import itertools

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
from gensim.models.doc2vec import Doc2Vec

from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec.load('./model')
# model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
# from gensim.test.utils import get_tmpfile
#
#
# fname = get_tmpfile("my_doc2vec_model")
#
# model.save(fname)
#
# model = Doc2Vec.load(fname)  # you can continue training with the loaded model!
# model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
# vector = model.infer_vector(["0.11", "0.31"])
# print(vector)


class IntegerSortDataset(Dataset):
	def __init__(self, num_samples=16, low=0, high=200, min_len=1, max_len=50, seed=1):

		self.prng = np.random.RandomState(seed=seed)
		self.input_dim = high

		self.data_root = '/home/huluwa/data/data_table'
		json_name_list = os.listdir(self.data_root)

		data_json_all = []
		for json_name in json_name_list:
			if '.json' in json_name:
				data_json_all.append(json_name)

		self.data_names = data_json_all[:980]
		# self.data_names = data_json_all[:76800]

		print('8888')
	def __getitem__(self, index):

		data_name = self.data_names[index]

		# data_train = []
		# data_label = []

		one_data = []
		one_label = []

		json_path = os.path.join(self.data_root, data_name)
		with open(json_path, 'r') as jf:
			data = json.load(jf)
			last_data = data[-1]
			last_box = last_data[2]
			bigx = last_box[2]
			bigy = last_box[3]

			for d in data:
				text = d[1]
				box = d[2]
				# box = d[2]
				parentid = d[5]
				# if ',' in d[3]:
				# 	parentid = int(d[3].split(',')[0])
				# else:
				# 	parentid = int(d[3])

				if parentid+2>49:
					one_label.append(0)
					one_data.append([0 for x in range(200)])
				else:
					one_label.append(parentid + 2)
					# box_ex = [x / 500 for x in box]
					x1 = box[0]/bigx
					y1 = box[1]/bigy
					x2 = box[2]/bigx
					y2 = box[3]/bigy
					box_ex = [x1,y1,x2,y2]
					vec = model.infer_vector(text.split()).transpose()
					vec_list = vec.tolist()
					one_data.append(box_ex*25+vec_list)


		train_extend = np.zeros((50, 200))
		len_train = len(one_data)
		one_train_np = np.array(one_data)[:50]
		train_extend[:len_train,:] = one_train_np


		label_extend = np.zeros(50)
		len_label = len(one_label)
		one_label_np = np.array(one_label)[:50]
		label_extend[:len_label] = one_label_np


		seq1 = train_extend
		label1 =label_extend

		len_seq1 = len(seq1)

		torch_tensor = torch.tensor(seq1)

		# return data, len_seq, label
		return torch_tensor, len_seq1, label1

	def __len__(self):
		return len(self.data_names)

def sparse_seq_collate_fn(batch):
	batch_size = len(batch)

	sorted_seqs, sorted_lengths, sorted_labels = zip(*sorted(batch, key=lambda x: x[1], reverse=True))

	padded_seqs = [seq.resize_as_(sorted_seqs[0]) for seq in sorted_seqs]

	# (Sparse) batch_size X max_seq_len X input_dim
	seq_tensor = torch.stack(padded_seqs)

	# batch_size
	length_tensor = torch.LongTensor(sorted_lengths)

	padded_labels = list(zip(*(itertools.zip_longest(*sorted_labels, fillvalue=-1))))

	# batch_size X max_seq_len (-1 padding)
	label_tensor = torch.LongTensor(padded_labels).view(batch_size, -1)

	# TODO: Currently, PyTorch DataLoader with num_workers >= 1 (multiprocessing) does not support Sparse Tensor
	# TODO: Meanwhile, use a dense tensor when num_workers >= 1.
	seq_tensor = seq_tensor.to_dense()

	return seq_tensor, length_tensor, label_tensor

def sparse_seq_collate_fn(batch):
	batch_size = len(batch)

	sorted_seqs, sorted_lengths, sorted_labels = zip(*sorted(batch, key=lambda x: x[1], reverse=True))

	padded_seqs = [seq.resize_as_(sorted_seqs[0]) for seq in sorted_seqs]

	# (Sparse) batch_size X max_seq_len X input_dim
	seq_tensor = torch.stack(padded_seqs)

	# batch_size
	length_tensor = torch.LongTensor(sorted_lengths)

	padded_labels = list(zip(*(itertools.zip_longest(*sorted_labels, fillvalue=-1))))

	# batch_size X max_seq_len (-1 padding)
	label_tensor = torch.LongTensor(padded_labels).view(batch_size, -1)

	# TODO: Currently, PyTorch DataLoader with num_workers >= 1 (multiprocessing) does not support Sparse Tensor
	# TODO: Meanwhile, use a dense tensor when num_workers >= 1.
	# seq_tensor = torch.LongTensor(seq_tensor)


	return seq_tensor.float(), length_tensor, label_tensor
