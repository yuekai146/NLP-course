from model import Summarization_Model
from batcher import *
from data import *
from utils import batch2var
from beam_search import run_beam_search



class hyper_params(object):
	def __init__(self, max_enc_steps, max_dec_steps, pointer_gen, batch_size, mode):
		self.max_enc_steps = max_enc_steps
		self.max_dec_steps = max_dec_steps
		self.pointer_gen = pointer_gen
		self.batch_size = batch_size
		self.mode = mode

def run_test(use_cuda=False):
	max_dec_steps = 100
	min_dec_step = 50
	beam_size = 20
	vocab = Vocab(
		'/home/zhaoyuekai/torch_code/data/summary/finished_files/vocab',
		50000
		)
	hps = hyper_params(50, 100, True, 20, 'encode')
	dataloader = Batcher(
		"/home/zhaoyuekai/torch_code/data/summary/finished_files/chunked/train_*"
		, vocab, hps, single_pass=True)
	net = Summarization_Model(
		50000, 128, 256, 400, unif_mag=0.02,trunc_norm_std=1e-4, 
		use_coverage=False, pointer_gen=True
		)
	if use_cuda:
		net = net.cuda()

	step = 0
	while step < 2:
		batch = dataloader.next_batch()
		batch = batch2var(batch, use_cuda)
		h = run_beam_search(
				net, vocab, batch, beam_size, max_dec_steps, min_dec_step, 
				use_cuda
			)
		print(len(h.tokens))
		print(h.tokens)
		step += 1