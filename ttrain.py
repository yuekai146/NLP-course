import argparse
import os
import shutil
import time
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from batcher import Batcher
import data
from utils import batch2var
from tmodel import Summarization_Model
from logger import Logger
import argparse
import data
from collections import namedtuple
import numpy as np

USE_CUDA=torch.cuda.is_available()

#2. set hyper parameters
parser = argparse.ArgumentParser(description='PyTorch Suaamrization Training')
parser.add_argument('--data-path', metavar='DIR', default="/home/zhaoyuekai/torch_code/data/summary/finished_files/chunked/train_*", help='Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
parser.add_argument('--vocab', metavar='DIR', default="/home/zhaoyuekai/torch_code/data/summary/finished_files/vocab",help='Path expression to text vocabulary file.')
# Where to find data
parser.add_argument('--log-dir', metavar='DIR',default="/home/zhaoyuekai/torch_code/summarization/NLP-course/logs/",help='where to save logger.')
parser.add_argument('--path-to-checkpoint', metavar='DIR',default="/home/zhaoyuekai/torch_code/summarization/NLP-course/checkpoints",help='where to save checkpoint.')
# Important settings
parser.add_argument('--mode', default='train', type=str, help='must be one of train/eval/decode')
parser.add_argument('--single-pass', type=bool, default=True, help='For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Hyperparameters
parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--extended-vsize', type=int, default=55000)
parser.add_argument('--emb-dim', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-enc-steps', type=int, default=400)
parser.add_argument('--max-dec-steps', type=int, default=100)
parser.add_argument('--min-dec-steps', type=int, default=35)
parser.add_argument('--vocab-size', type=int, default=50000)
parser.add_argument('--lr', type=float, default=0.15)
parser.add_argument('--adagrad-init-acc', type=float, default=0.1)
parser.add_argument('--rand-unif-init-mag', type=float, default=0.02)
parser.add_argument('--trunc-norm-init-std', type=float, default=1e-4)
parser.add_argument('--max-grad-norm', type=float, default=2.0)
parser.add_argument('--num-steps', type=int, default=3000)
parser.add_argument('--check-n', type=int, default=5)


# Pointer-generator or baseline model
parser.add_argument('--pointer-gen',default=True,type=bool,help='If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
parser.add_argument('--coverage', default=False, type=bool,help='Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
parser.add_argument('--cov-loss-wt', default=1.0, type=float,help='Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
parser.add_argument('--convert-to-coverage-model', default=False, type=bool, help='Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
parser.add_argument('--restore-best-model', default=False, type=bool,help='Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging.
parser.add_argument('--debug',default=False)

args = parser.parse_args()


#6. write some utility functions
# 	6.1 save checkpoint per N batches
#is_best是什么？？？
def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
	torch.save(state,filename)
	if is_best:
		shutil.copyfile(filename,'model_best.pth.tar')
# 	6.2 logger to illustrate loss
# 		>>> logger = Logger(**args)
# 		>>> logger.scalar_summary(**args)

logger = Logger(args.log_dir)
def log(tag, value, step):
	logger.scalar_summary(tag, value, step)

# 	6.3 loss function (MLE loss and coverage loss)
def loss_function(final_dists,atten_dists,batch):
	if args.pointer_gen:
		loss_per_step=[]
		for dec_step, dist in enumerate(final_dists):
			# print(dist.numpy()[0])
			targets = batch.target_batch[:,dec_step].contiguous().view(-1, 1)
			targets_ = targets.data
			indices = torch.arange(args.batch_size).view(-1, 1).long()
			
			#targets = targets.data.numpy()
			# print("targets:", len(targets))
			# print(type(targets))
			# indices = np.stack((batch_nums,targets), axis=1)##
			# indices = torch.LongTensor(indices)
			# print(indices)
			# batch_n=0
			# while batch_n<args.batch_size:
			# 	gold_probs = []
			# 	gold_probs.append(dist[batch_n][targets[batch_n]])
			# 	batch_n += 1
			# gold_probs = torch.gather(dist, indices)
			gold_probs = dist[indices, targets_]
			# print(gold_probs)
			losses = -torch.log(gold_probs)
			loss_per_step.append(losses)
		loss = _mask_and_avg(loss_per_step, batch.dec_padding_mask)
	if args.coverage:
		coverage_loss = _coverage_loss(atten_dists, batch.dec_padding_mask)
		loss = loss + args.cov_loss_wt*coverage_loss
	return loss






def _mask_and_avg(values, padding_mask):
	"""Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  	"""
	dec_lens = torch.sum(padding_mask,dim=1)
	values_per_step = [v*padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
	values_per_ex = sum(values_per_step)/dec_lens
	return torch.sum(values_per_ex)
def _coverage_loss(atten_dists, padding_mask):
	"""Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
	"""
	coverage = torch.zeros_like(attn_dists[0])
	covlosses = []
	for a in attn_dists:
		covloss = torch.sum(torch.min(a,coverage), dim=1)
		covlosses.append(covloss)
		coverage += a
	coverage_loss = _mask_and_avg(covlosses, padding_mask)
	return coverage_loss
'''

7. start training
	>>> batch = next(dataloader)
	>>> batch = batch2var(batch)
	>>> final_dists, attn_dists = net(batch, use_cuda=True)
	>>> loss = loss_function(final_dists, attn_dists, batch)
	>>> optimizer.zero_grad()
	>>> loss.backward()
	>>> optimizer.step()
	>>> logger
	>>> save_checkpoint()
'''
#args.N取多少？？？
vocab = data.Vocab(args.vocab, args.vocab_size)
hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
hps_dict = {'mode':args.mode, 'lr':args.lr, 'adagrad_init_acc':args.adagrad_init_acc,
'rand_unif_init_mag':args.rand_unif_init_mag, 'trunc_norm_init_std':args.trunc_norm_init_std, 'max_grad_norm':args.max_grad_norm,
'hidden_dim':args.hidden_dim, 'emb_dim':args.emb_dim, 'batch_size':args.batch_size, 'max_dec_steps':args.max_dec_steps,
'max_enc_steps':args.max_enc_steps, 'coverage':args.coverage, 'cov_loss_wt':args.cov_loss_wt, 'pointer_gen':args.pointer_gen}

hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

dataloader = Batcher(args.data_path, vocab, hps, args.single_pass)
# print(dataloader)


def forward(attn_len, max_dec_steps, batch_size, extended_vsize, use_cuda=False,mode='train', pointer_gen=True, use_coverage=False):
	final_lists=[]
	for i in range(max_dec_steps):
		fd = Variable(torch.rand(batch_size, extended_vsize))
		if use_cuda:
			fd = fd.cuda()
		final_lists.append(fd)
	attn_lists = []
	for j in range(max_dec_steps):
		ad = Variable(torch.rand(batch_size, attn_len))
		if use_cuda:
			ad = ad.cuda()
		attn_lists.append(ad)
	return final_lists, attn_lists
final_dists, attn_dists = forward(attn_len=args.max_enc_steps, max_dec_steps=args.max_dec_steps, batch_size=args.batch_size, extended_vsize=args.extended_vsize, use_cuda=False, mode=args.mode, pointer_gen=args.pointer_gen, use_coverage=args.coverage)




for n in range(args.num_steps):
	print("lalallalalallalalallalalallalallalla:",n)
	batch = dataloader.next_batch()
	batch = batch2var(batch, use_cuda=False)
	loss = loss_function(final_dists, attn_dists, batch)
	print(loss)
	log('loss', loss.data[0], step=n)
	if n % args.check_n == 0:
		save_checkpoint({
			'step': n + 1},
			is_best=False)
	if n > 20:
		break