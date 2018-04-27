from batcher import *
class hyper_params(object):
	def __init__(self, max_enc_steps, max_dec_steps, pointer_gen, batch_size, mode):
		self.max_enc_steps = max_enc_steps
		self.max_dec_steps = max_dec_steps
		self.pointer_gen = pointer_gen
		self.batch_size = batch_size
		self.mode = mode

vocab = data.Vocab("/home/zhaoyuekai/torch_code/data/summary/finished_files/vocab", 50000)
hps = hyper_params(800, 100, True, 20, 'encode')
dataloader = Batcher("/home/zhaoyuekai/torch_code/data/summary/finished_files/chunked/train_*", vocab, hps, True)
batch = dataloader.next_batch()
print(batch.target_batch)
print("\n\n")
print(batch.dec_batch)
print("\n\n")
print(batch.enc_batch_extend_vocab)
print("\n\n")
print(batch.enc_batch)