'''
1. import all packages needed

2. set hyper parameters

3. Initialize a dataloader
	>>> vocab = data.Vocab(**args)
	>>> dataloader = Batcher(**args)
	>>> batch = next(dataloader)
	>>> batch = batch2var(batch, use_cuda=True)

4. Initialize a net or store from checkpoint
	>>> net = Summarization_Model(**args)

	>>> net = torch.load("path_to_checkpoint")

5. Initlialize a optimizer
	>>> optimizer = torch.optim.Adagrad(**args)

6. write some utility functions
	6.1 save checkpoint per N batches

	6.2 logger to illustrate loss
		>>> logger = Logger(**args)
		>>> logger.scalar_summary(**args)

	6.3 loss function (MLE loss and coverage loss)

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