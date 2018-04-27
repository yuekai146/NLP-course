from torch.autograd import Variable
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_Decoder(nn.Module):

	def __init__(self, embed_dim, hidden_dim, attn_len, unif_mag=None):
		'''
		Args:
			embed_dim: word embedding dimension.
			hidden_dim: hidden dimension of LSTM cell.
			attn_len: number of encoder hidden state to attend to.
			unif_mag: magnitude of uniform distributuin.
		'''
		super(Attention_Decoder, self).__init__()
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.attn_len = attn_len
		self.unif_mag = unif_mag

		self.W_s = nn.Linear(2 * hidden_dim, hidden_dim)
		self.input2x = nn.Linear(2 * hidden_dim + embed_dim, embed_dim)
		self.W_p_gen = nn.Linear(4 * hidden_dim + embed_dim, 1)
		self.W_out = nn.Linear(3 * hidden_dim, hidden_dim)

		self.W_h = nn.Conv2d(2 * hidden_dim, hidden_dim, 1)
		self.W_c = nn.Conv2d(1, hidden_dim, 1)
		self.v = nn.Parameter(torch.Tensor(hidden_dim))
		self.cell = nn.LSTM(
			embed_dim, hidden_dim, num_layers=1, batch_first=True
			)
		self._init_params_()

	def _init_params_(self):
		# May need more initialization operations!
		nn.init.uniform(self.v)
		if self.unif_mag != None:
			for param in self.cell.parameters():
				nn.init.uniform(param, a=-self.unif_mag, b=self.unif_mag)

	def decode_onestep(self, step, decoder_input, dec_in_state, encoder_states,
					   encoder_features, enc_padding_mask, pointer_gen=True,
					   use_coverage=False, prev_coverage=None, use_cuda=False,
					   initial_state_attention=False):
		'''
		Args:
			decoder_input: A 2-D tensor [batch_size * embed_size].
			initial_state: 2-D tensor [batch_size * hidden_dim].
			encoder_states: 3-D tensor [batch_size * attn_len * 2 * hidden_dim].
			enc_padding_mask: 2-D tensor [batch_size * attn_len].
							  Only contains 0s and 1s.
							  0 for paddings and 1 for real tokens.
			initial_state_attention: Only True for test.
			pointer_gen: Whether to use pointer generator mechanism.
			use_coverage: Whetherto use coverage mechanism.
			prev_coverage: If not None, a 2-D tensor [batch_size * attn_len].
		'''
		batch_size, attn_len, _ = encoder_states.size()
		
		encoder_states = torch.unsqueeze(encoder_states, dim=2)

		if prev_coverage is not None:
			prev_coverage = torch.unsqueeze(
				torch.unsqueeze(prev_coverage, 2), 3
				)
			# Turn to [batch_size * attn_len * 1 * 1]
		def attention(decoder_state, coverage=None):
			"""
				Calculate the context vector and attention distribution 
				from the decoder state.

		        Args:
		          decoder_state: state of the decoder
		          coverage: Optional. Previous timestep's coverage vector, 
		        		    shape (batch_size, attn_len, 1, 1).

		        Returns:
		          context_vector: weighted sum of encoder_states
		          attn_dists: attention distribution
		          coverage: new coverage vector.
		          			Shape (batch_size, attn_len, 1, 1)
	      	"""
			decoder_features = self.W_s(torch.cat(decoder_state, dim=1))
			decoder_features = torch.unsqueeze(
      					torch.unsqueeze(decoder_features, 1), 1
      					)
			def masked_attention(e):
				# e is a 2-D tensor [batch_size * attn_len]
				attn_dist = F.softmax(e, dim=1)
				attn_dist *= enc_padding_mask.float()
				masked_sums = torch.sum(attn_dist, dim=1)
				attn_dist /= torch.unsqueeze(masked_sums, dim=1)
				return attn_dist
			if use_coverage and coverage is not None:
				coverage_features = self.W_c(coverage.permute(0, 3, 1, 2))
				coverage_features = coverage_features.permute(0, 2, 3, 1)
				# shape [batch_size * attn_len * 1 * hidden_dim]
				e = torch.sum(self.v * F.tanh(
					encoder_features + decoder_features + coverage_features
					), dim=2)
				e = torch.sum(e, dim=2)
				# e is a 2-D tensor [batch_size * attn_len]
				attn_dist = masked_attention(e)

				coverage += attn_dist.resize(batch_size, -1, 1, 1)
			else:
				e = torch.sum(self.v * F.tanh(
					encoder_features + decoder_features
					), dim=2)
				e = torch.sum(e, dim=2)

				attn_dist = masked_attention(e)

				if use_coverage:
					coverage = torch.unsqueeze(
						torch.unsqueeze(attn_dist, 2), 2
						)
			context_vector = torch.sum(
				torch.unsqueeze(
					torch.unsqueeze(attn_dist, 2), 3
					) * encoder_states, dim=1
				)
			context_vector = torch.sum(context_vector, dim=1)
			# context_vector is of size [batch_size * (2 * hidden_dim)]

			return context_vector, attn_dist, coverage

		h, c = dec_in_state
		h = torch.unsqueeze(h, dim=0)
		c = torch.unsqueeze(c, dim=0)
		state = (h, c)
		coverage = prev_coverage
		context_vector = Variable(torch.zeros(batch_size, 2 * self.hidden_dim))
		if use_cuda:
			context_vector = context_vector.cuda()

		if initial_state_attention:
			context_vector, _, coverage = attention(dec_in_state, coverage)
		
		x = self.input2x(
			torch.cat([decoder_input, context_vector], dim=1)
			)
		x_ = torch.unsqueeze(x, dim=1)

		cell_output, (h, c) = self.cell(x_, state)
		cell_output = torch.squeeze(cell_output)
		h = torch.squeeze(h)
		c = torch.squeeze(c)
		state = (h, c)

		if step == 0 and initial_state_attention:
			context_vector, attn_dist, _ = attention(state, coverage)
		else:
			context_vector, attn_dist, coverage = attention(state, coverage)

		if pointer_gen:
			c, h = state
			p_gen = self.W_p_gen(
				torch.cat([context_vector, c, h, x], dim=1)
				)
			p_gen = F.sigmoid(p_gen)

		output = self.W_out(torch.cat([cell_output, context_vector], dim=1))

		if coverage is not None:
			coverage = torch.squeeze(coverage)
		return output, state, attn_dist, p_gen, coverage
	
	def forward(self, decoder_inputs, initial_state, encoder_states,
				enc_padding_mask, initial_state_attention=False,
				pointer_gen=True, use_coverage=False, prev_coverage=None,
				use_cuda=False):
		'''
		Args:
			decoder_inputs: A list of length max_dec_steps.
							Each element is a 2-D tensor.
							Tensor shape is [batch_size * embed_size].
			initial_state: 2-D tensor [batch_size * hidden_dim].
			encoder_states: 3-D tensor [batch_size * attn_len * 2 * hidden_dim].
			enc_padding_mask: 2-D tensor [batch_size * attn_len].
							  Only contains 0s and 1s.
							  0 for paddings and 1 for real tokens.
			initial_state_attention: Only True for test.
			pointer_gen: Whether to use pointer generator mechanism.
			use_coverage: Whetherto use coverage mechanism.
			prev_coverage: If not None, a 2-D tensor [batch_size * attn_len].
		'''
		
		encoder_states_ = torch.unsqueeze(encoder_states, dim=2)
		encoder_features = self.W_h(encoder_states_.permute(0, 3, 1, 2))
		encoder_features = encoder_features.permute(0, 2, 3, 1)

		outputs = []
		attn_dists = []
		p_gens = []
		state = initial_state
		
		for i, inp in enumerate(decoder_inputs):
			output, state, attn_dist, p_gen, coverage = self.decode_onestep(
				i, inp, state, encoder_states, encoder_features,
				enc_padding_mask, pointer_gen, use_coverage,
				prev_coverage, use_cuda, initial_state_attention  
				)
			attn_dists.append(attn_dist)

			if pointer_gen:
				p_gens.append(p_gen)

			outputs.append(output)
		return outputs, state, attn_dists, p_gens, coverage

class Summarization_Model(nn.Module):

	def __init__(self, vocab_size, embed_dim, hidden_dim, attn_len,
				 num_layers=1, mode='train', unif_mag=None,
				 trunc_norm_std=None, pointer_gen=True,
				 initial_state_attention=False, use_coverage=False,
				 prev_coverage=None):
		super(Summarization_Model, self).__init__()
		# Initialize all relevant parameters.
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.attn_len = attn_len
		self.num_layers = num_layers
		self.mode = mode
		self.unif_mag = unif_mag
		self.trunc_norm_std = trunc_norm_std
		self.initial_state_attention = initial_state_attention
		self.pointer_gen = pointer_gen
		self.use_coverage = use_coverage
		if mode == 'decode' and prev_coverage:
			self.prev_coverage = prev_coverage
		else:
			self.prev_coverage = None


		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.encoder = nn.LSTM(
				input_size=embed_dim, hidden_size=hidden_dim,
				num_layers=num_layers, batch_first=True,
				bidirectional=True
				)
		self.reduce_state_c = nn.Linear(2 * num_layers * hidden_dim, hidden_dim)
		self.reduce_state_h = nn.Linear(2 * num_layers * hidden_dim, hidden_dim)

		self.decoder = Attention_Decoder(
					embed_dim, hidden_dim,attn_len, unif_mag
					)
		self.output_proj = nn.Linear(hidden_dim, vocab_size)
		self._init_params_()

	def _init_params_(self):
		# May need more initialization.

		for param in self.reduce_state_c.parameters():
			nn.init.normal(param, std=self.trunc_norm_std)

		for param in self.reduce_state_h.parameters():
			nn.init.normal(param, std=self.trunc_norm_std)

		for param in self.output_proj.parameters():
			nn.init.normal(param, std=self.trunc_norm_std)

	def prob_output_onestep(self, decoder_output, attn_dist, p_gen, batch,
							use_cuda):
		batch_size, seq_len = batch.enc_batch.size()
		# 5. Use decoder outputs to compute vocab_dists
		vocab_score = self.output_proj(decoder_output)
		vocab_dist = F.softmax(vocab_score, dim=1)

		# 6. Use p_gen, vocab_dists and attn_dists to compute final_dists
		if self.pointer_gen:
			inputs = batch.enc_batch_extend_vocab.cpu().data.numpy()
			numbers = inputs.reshape(-1).tolist()
			set_numbers = list(set(numbers))
			if 1 in set_numbers:
				set_numbers.remove(1)
			c = Counter(numbers)
			dup_list = [k for k in set_numbers]

			def compute_final_dist(attn, vocab_dist, p_gen):
				attn_sum_list = []
				for dup in dup_list:
					mask = np.array(inputs == dup, dtype=float)
					mask = Variable(torch.from_numpy(mask)).float()
					if use_cuda:
						mask = mask.cuda()
					attn_mask = torch.mul(mask, attn)
					attn_sum = attn_mask.sum(1).unsqueeze(1)
					attn_sum_list.append(attn_sum)
				attn_sums = torch.cat(attn_sum_list, dim=1)
					
				p_copy = torch.zeros(
					batch_size, self.vocab_size + batch.max_art_oovs
					)
				p_copy = Variable(p_copy)
				if use_cuda:
					p_copy = p_copy.cuda()
				
				'''
				print("attn size = {}".format(attn.size()))
				print("batch_indices:\n")
				print(batch_indices)
				print("word_indices:\n")
				print(word_indices)
				print("idx_repeat:\n")
				print(idx_repeat)
				'''
				for i, k in enumerate(dup_list):
					p_copy[:, k] = attn_sums[:, i]
				extend = Variable(torch.zeros(batch_size, batch.max_art_oovs))
				if use_cuda:
					extend = extend.cuda()
				vocab_dist_ = torch.cat([vocab_dist, extend], dim=1)
				p_out = torch.mul(vocab_dist_.t(), torch.squeeze(p_gen)) + \
						torch.mul(p_copy.t(), (1-torch.squeeze(p_gen)))
				p_out = p_out.t()
				return p_out

			final_dist = compute_final_dist(attn_dist, vocab_dist, p_gen)
		else:
			final_dist = vocab_dist
	
		# 7. Output final_dists for MLE loss and attn_dists for coverage loss
		return final_dist

	def forward(self, batch, use_cuda=False):
		'''
		Args:
			batch: Contains following information we need
				   (1) dec_batch [batch_size * max_dec_steps]
				   (2) target_batch [batch_size * max_dec_steps]
				   (3) dec_padding_mask [batch_size * max_dec_steps]
				   (4) enc_batch [batch_size * max_enc_steps]
				   (5) enc_lens [batch_size]
				   (6) enc_input_extend_vocab [batch_size * max_enc_steps]
				   (7) enc_padding_mask [batch_size * max_enc_steps]
			use_cuda: Wheteher to perform calculation in GPU.
		'''
		batch_size, seq_len = batch.enc_batch.size()
		# 1. Embed enc_batch and dec_batch
		emb_enc_inputs = self.embedding(batch.enc_batch)
		emb_dec_inputs = [torch.squeeze(self.embedding(x)) for x in torch.split(
					batch.dec_batch, 1, dim=1
				)]
		# 2. Compute encoder_outputs, h_t, c_t
		# hidden_outputs are of size [batch_size * seq_len * (2 * hidden_dim)]
		h_0 = Variable(
			torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim)
			)
		c_0 = Variable(
			torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim)
			)
		if use_cuda:
			h_0 = h_0.cuda()
			c_0 = c_0.cuda()
		encoder_outputs, (h_t, c_t) = self.encoder(emb_enc_inputs, (h_0, c_0))

		# 3. Compute initial state for decoder
		# new_h and new_c are of size [batch_size * hidden_dim]
		new_c = self.reduce_state_c(
			c_t.permute(1, 0, 2).contiguous().view(batch_size, -1)
			)
		new_h = self.reduce_state_h(
			h_t.permute(1, 0, 2).contiguous().view(batch_size, -1)
			)

		# 4. Compute decoder outputs attn_dists and so on.
		decoder_outputs, decoder_out_state, attn_dists, p_gens, coverage = \
		self.decoder(emb_dec_inputs, (new_h, new_c), encoder_outputs,
				batch.enc_padding_mask, self.initial_state_attention,
				self.pointer_gen, self.use_coverage, self.prev_coverage,
				use_cuda)

	
		final_dists = []
		for decoder_output, attn_dist, p_gen in zip(
				decoder_outputs, attn_dists, p_gens):
			final_dists.append(
				self.prob_output_onestep(
					decoder_output, attn_dist, p_gen, batch, use_cuda
					)
				)
	
		# 7. Output final_dists for MLE loss and attn_dists for coverage loss
		return final_dists, attn_dists