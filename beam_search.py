from torch.autograd import Variable
import data
import numpy as np
import torch


class Hypothesis(object):

	def __init__(self, tokens, log_probs, state, coverage):
		self.tokens = tokens
		self.log_probs = log_probs
		self.state = state
		self.coverage = coverage

	def extend(self, token, log_prob, state, coverage):
		return Hypothesis(
			self.tokens + token, self.log_probs + log_prob, state, coverage
			)

	@property
	def avg_log_prob(self):
		return sum(self.log_probs) / len(self.tokens)

def get_log_prob(batch, use_cuda, net, token, prev_decoder_state,
				 encoder_states, encoder_features, coverage,
				 beam_size, step):
	'''
	Args:
		batch: A batch object containing test example to summarize.
		net: A trained network.
		token: A 1-D longtensor [batch_size].
		prev_decode_state: Decoder hidden state.
		emb_enc_inputs: Embedded encoder inputs.
	'''
	batch_size, seq_len = batch.enc_batch.size()
	# 1. Embed the token
	emb_dec_input = net.embedding(token) # 2-D tensor [batch_size * embed_dim]
	# 2. Perform one step decoding.
	state = prev_decoder_state

	output, state, attn_dist, p_gen, coverage = net.decoder.decode_onestep(
		step, emb_dec_input, state, encoder_states, encoder_features,
		batch.enc_padding_mask, net.pointer_gen, net.use_coverage,
		coverage, use_cuda, net.initial_state_attention  
		)

	# 3. Compute final distribution and top k log probability.
	final_dist = net.prob_output_onestep(
				output, attn_dist, p_gen, batch, use_cuda
			)
	topk_probs, topk_tokens = torch.topk(final_dist, 2 * beam_size, dim=1)
	topk_log_probs = torch.log(topk_probs)
	return topk_log_probs, topk_tokens, state, coverage

def run_beam_search(net, vocab, batch, beam_size, max_dec_steps, min_dec_step, 
					use_cuda):
	batch_size, seq_len = batch.enc_batch.size()

	# 1. Embed enc_batch.
	emb_enc_inputs = net.embedding(batch.enc_batch)

	# 2. Compute encoder_outputs, h_t, c_t
	# hidden_outputs are of size [batch_size * seq_len * (2 * hidden_dim)]
	h_0 = Variable(
		torch.zeros(2 * net.num_layers, batch_size, net.hidden_dim)
		)
	c_0 = Variable(
		torch.zeros(2 * net.num_layers, batch_size, net.hidden_dim)
		)
	if use_cuda:
		h_0 = h_0.cuda()
		c_0 = c_0.cuda()
	encoder_states, (h_t, c_t) = net.encoder(emb_enc_inputs, (h_0, c_0))

	# 3. Compute initial state for decoder
	# new_h and new_c are of size [batch_size * hidden_dim]
	new_c = net.reduce_state_c(
		c_t.permute(1, 0, 2).contiguous().view(batch_size, -1)
		)
	new_h = net.reduce_state_h(
		h_t.permute(1, 0, 2).contiguous().view(batch_size, -1)
		)

	encoder_states_ = torch.unsqueeze(encoder_states, dim=2)
	encoder_features = net.decoder.W_h(encoder_states_.permute(0, 3, 1, 2))
	encoder_features = encoder_features.permute(0, 2, 3, 1)

	initial_coverage = Variable(torch.zeros(batch_size, seq_len))
	if use_cuda:
		initial_coverage = initial_coverage.cuda()
	coverage = initial_coverage

	# 4.Perform one step decoding.
	hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
					   log_probs=[0.0],
					   state=(new_h[i], new_c[i]),
					   coverage=initial_coverage[i]) for i in range(beam_size)]
	step = 0
	results = []

	while step < max_dec_steps and len(results) < beam_size:
		latest_tokens = [h.tokens[-1] for h in hyps]
		latest_tokens = [
		t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) \
		for t in latest_tokens
						 ]
		latest_tokens = Variable(torch.from_numpy(np.array(latest_tokens)))
		latest_tokens = latest_tokens.long()
		if use_cuda:
			latest_tokens = latest_tokens.cuda()
		state_h = [h.state[0] for h in hyps]
		state_c = [h.state[1] for h in hyps]
		state_h = torch.stack(state_h, dim=0)
		state_c = torch.stack(state_c, dim=0)

		topk_log_probs, topk_tokens, state, coverage = get_log_prob(
			batch, use_cuda, net, latest_tokens, (state_h, state_c),
			encoder_states, encoder_features, coverage,
			beam_size, step
			)
		topk_log_probs = topk_log_probs.data.cpu().numpy()
		topk_tokens = topk_tokens.data.cpu().numpy()
		all_hyps = []
		num_orig_hyps = 1 if step == 0 else len(hyps)

		for i in range(num_orig_hyps):
			h = hyps[i]
			new_state_h = state[0][i]
			new_state_c = state[1][i]
			new_coverage = coverage[i]
			for j in range(beam_size):
				new_hyp = h.extend(token=[topk_tokens[i, j]],
								   log_prob=[topk_log_probs[i, j]],
								   state=(new_state_h, new_state_c),
								   coverage=new_coverage)
				all_hyps.append(new_hyp)

		hyps = []
		for h in sort_hyps(all_hyps):
			if h.tokens[-1] == vocab.word2id(data.STOP_DECODING):
				if steps >= min_dec_step:
					results.append(h)
			else:
				hyps.append(h)
				if len(hyps) == beam_size or len(results) == beam_size:
					break
		step += 1
	if len(results) == 0:
		results = hyps

	hyps_sorted = sort_hyps(results)

	return hyps_sorted[0]

def sort_hyps(hyps):
	return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)