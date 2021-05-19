import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader
import os

from dataset_refined_ocr_test import IntegerSortDataset, sparse_seq_collate_fn
from model_330 import PointerNet
parser = argparse.ArgumentParser(description='PtrNet-Sorting-Integer')

parser.add_argument('--low', type=int, default=0, help='lowest value in dataset (default: 0)')
parser.add_argument('--high', type=int, default=200, help='highest value in dataset (default: 100)')
parser.add_argument('--min-length', type=int, default=4, help='minimum length of sequences (default: 5)')
parser.add_argument('--max-length', type=int, default=50, help='maximum length of sequences (default: 20)')
parser.add_argument('--train-samples', type=int, default=14, help='number of samples in train set (default: 100000)')
parser.add_argument('--test-samples', type=int, default=14, help='number of samples in test set (default: 1000)')

parser.add_argument('--emb-dim', type=int, default=128, help='embedding dimension (default: 8)')
parser.add_argument('--batch-size', type=int, default=2, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100000, help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay (default: 1e-5)')

parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

args = parser.parse_args()
test_set = IntegerSortDataset(num_samples=args.test_samples, high=args.high, min_len=args.min_length,
                              max_len=args.max_length, seed=2)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                         collate_fn=sparse_seq_collate_fn)


use_cuda = not args.no_cuda and torch.cuda.is_available()
print('use_cuda',use_cuda)
device = torch.device("cuda")
print('device',device)
model = PointerNet(input_dim=args.high, embedding_dim=256, hidden_size=256).to(device)

model.load_state_dict(torch.load('/home/huluwa/data/final_model/newmodel_mother_041.pth'))
model.eval()



txt_root = '/home/huluwa/data/data_final/m2'
k = 0

for seq, length, name in test_loader:
    seq, length = seq.to(device), length.to(device)

    log_pointer_score, argmax_pointer, mask = model(seq, length)
    txt_path = os.path.join(txt_root, str(k)+'.txt')
    print('predi',argmax_pointer)
    # print('label:',name)
    # predict.data.cpu().numpy()
    result1 = np.array(argmax_pointer.float().cpu())
    np.savetxt(txt_path, result1)

    k = k + 1
