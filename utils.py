from torch.autograd import Variable
import torch


def batch2var(batch, use_cuda=False):
	# When training, first sample a batch from an instance of the batcher class
	# then call batch2var(batch) before feed the batch to the net
	# e.g.
	#        >>> batch = next(dataloader)
	#		 >>> batch = batch2var(batch)
	#  		 >>> final_dists, attn_dists = net(batch, use_cuda=True) 
	batch.enc_batch = Variable(torch.from_numpy(batch.enc_batch)).long()
	batch.enc_lens = Variable(torch.from_numpy(batch.enc_lens)).long()
	batch.enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask))
	batch.enc_padding_mask = batch.enc_padding_mask.long()
	batch.enc_batch_extend_vocab = Variable(
		torch.from_numpy(batch.enc_batch_extend_vocab)
		).long()

	batch.dec_batch = Variable(torch.from_numpy(batch.dec_batch)).long()
	batch.target_batch = Variable(torch.from_numpy(batch.target_batch)).long()
	batch.dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask))
	batch.dec_padding_mask = batch.dec_padding_mask.float()

	if use_cuda:
		batch.enc_batch = batch.enc_batch.cuda()
		batch.enc_lens = batch.enc_lens.cuda()
		batch.enc_padding_mask = batch.enc_padding_mask.cuda()
		batch.enc_batch_extend_vocab = batch.enc_batch_extend_vocab.cuda()

		batch.dec_batch = batch.dec_batch.cuda()
		batch.target_batch = batch.target_batch.cuda()
		batch.dec_padding_mask = batch.dec_padding_mask.cuda()
	return batch