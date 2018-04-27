from model import Summarization_Model
from batcher import *
from data import *
from utils import batch2var



class hyper_params(object):
	def __init__(self, max_enc_steps, max_dec_steps, pointer_gen, batch_size, mode):
		self.max_enc_steps = max_enc_steps
		self.max_dec_steps = max_dec_steps
		self.pointer_gen = pointer_gen
		self.batch_size = batch_size
		self.mode = mode

def run_test(use_cuda=False):
	vocab = Vocab(
		'/home/zhaoyuekai/torch_code/data/summary/finished_files/vocab',
		50000
		)
	hps = hyper_params(50, 100, True, 20, 'encode')
	dataloader = Batcher(
		"/home/zhaoyuekai/torch_code/data/summary/finished_files/chunked/train_*"
		, vocab, hps, single_pass=True)
	net = Summarization_Model(
		50000, 128, 256, 50, unif_mag=0.02,trunc_norm_std=1e-4, 
		use_coverage=True
		)
	if use_cuda:
		net = net.cuda()

	step = 0
	while step < 2:
		batch = dataloader.next_batch()
		batch = batch2var(batch, use_cuda)
		final_dists, attn_dists = net(batch, use_cuda)
		print("Final distribution is of length:{}".format(len(final_dists)))
		print("Attention is of length:{}".format(len(attn_dists)))
		print("Final distribution is of size:{}".format(final_dists[1].size()))
		print("Attention is of size:{}".format(attn_dists[1].size()))
		step += 1

run_test(True)