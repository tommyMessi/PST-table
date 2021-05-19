import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from dataset_refined_ocr import IntegerSortDataset, sparse_seq_collate_fn
from model_330 import PointerNet

parser = argparse.ArgumentParser(description='PtrNet-Sorting-Integer')

parser.add_argument('--low', type=int, default=0, help='lowest value in dataset (default: 0)')
parser.add_argument('--high', type=int, default=200, help='highest value in dataset (default: 100)')
parser.add_argument('--min-length', type=int, default=5, help='minimum length of sequences (default: 5)')
parser.add_argument('--max-length', type=int, default=50, help='maximum length of sequences (default: 20)')
parser.add_argument('--train-samples', type=int, default=9600, help='number of samples in train set (default: 100000)')
parser.add_argument('--test-samples', type=int, default=10, help='number of samples in test set (default: 1000)')

parser.add_argument('--emb-dim', type=int, default=128, help='embedding dimension (default: 8)')
parser.add_argument('--batch-size', type=int, default=2, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100000, help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 1e-3)')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-5)')

parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def masked_accuracy(output, target, mask):
	"""Computes a batch accuracy with a mask (for padded sequences) """
	with torch.no_grad():
		masked_output = torch.masked_select(output, mask)
		masked_target = torch.masked_select(target, mask)
		accuracy = masked_output.eq(masked_target).float().mean()

		return accuracy


def main():
	args = parser.parse_args()

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda")
	cudnn.benchmark = True



	train_set = IntegerSortDataset(num_samples=args.train_samples, high=args.high, min_len=args.min_length, max_len=args.max_length, seed=1)
	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=sparse_seq_collate_fn)

	test_set = IntegerSortDataset(num_samples=args.test_samples, high=args.high, min_len=args.min_length, max_len=args.max_length, seed=2)
	test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=sparse_seq_collate_fn)
								#100
	model = PointerNet(input_dim=args.high, embedding_dim=256, hidden_size=256).to(device)
	optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1, last_epoch=-1)
	train_loss = AverageMeter()
	train_accuracy = AverageMeter()
	test_loss = AverageMeter()
	test_accuracy = AverageMeter()

	for epoch in range(args.epochs):
		# Train
		scheduler.step()
		model.train()
		for batch_idx, (seq, length, target) in enumerate(train_loader):

			seq, length, target = seq.to(device), length.to(device), target.to(device)

			optimizer.zero_grad()
			log_pointer_score, argmax_pointer, mask = model(seq, length)

			unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
			loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
			assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

			loss.backward()
			optimizer.step()

			train_loss.update(loss.item(), seq.size(0))

			mask = mask[:, 0, :]
			train_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())

			if batch_idx % 20 == 0:
				# torch.save(model, './best90post.pth')
				# torch.save(model.state_dict(), 't.pth')

				print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
					  .format(epoch, batch_idx * len(seq), len(train_loader.dataset),
							  10. * batch_idx / len(train_loader), train_loss.avg, train_accuracy.avg))

		if epoch % 1 == 0:


			torch.save(model.state_dict(), 'data_father_0415.pth')




if __name__ == '__main__':
	main()
