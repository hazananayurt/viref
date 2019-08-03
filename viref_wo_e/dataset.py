import os
import pickle
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class VirefDataset(Dataset):

	def __init__(self, args, refexp_csv, max_refexp_len=25, video_name_restriction=None):
		self.refexp_list = []
		self.video_names = set([])
		
		self.word_embed = {}
		self.word2idx = {}
		self.max_refexp_len = max_refexp_len
		self.video_name_restriction = video_name_restriction
		
		self.object_features = {}
		self.object_vgg_features = {}
		self.pair_features = {}
		self.pair_features_blackened = {}
		
		features_dir = args.features_dir
		dataset_split_dir = args.dataset_split_dir
		data_dir = args.data_dir
		
		with open(refexp_csv, newline='') as csv_file:
			reader = csv.reader(csv_file)
			for video_name, obj1, obj2, refexp in reader:
				if len(refexp.split()) > self.max_refexp_len:
					continue
				if self.video_name_restriction is not None and self.video_name_restriction != video_name:
					continue
				self.refexp_list.append([video_name, int(obj1), int(obj2), refexp])
		
		for video_name, obj1, obj2, refexp in self.refexp_list:
			self.video_names.add(video_name)
		
		for video_name, obj1, obj2, refexp in self.refexp_list:
			obj1, obj2 = sorted([obj1, obj2])
			obj1_feature_path = os.path.join(features_dir, 'object_blackened', '{:s}__{:d}'.format(video_name, obj1)) #c3d blackened features
			obj2_feature_path = os.path.join(features_dir, 'object_blackened', '{:s}__{:d}'.format(video_name, obj2))
			obj1_vgg_feature_path = os.path.join(features_dir, 'object', '{:s}__{:d}'.format(video_name, obj1))
			obj2_vgg_feature_path = os.path.join(features_dir, 'object', '{:s}__{:d}'.format(video_name, obj2))
			pair_feature_path = os.path.join(features_dir, 'pair', '{:s}__{:d}_{:d}'.format(video_name, obj1, obj2))
			pair_feature_blackened_path = os.path.join(features_dir, 'pair_blackened', '{:s}__{:d}_{:d}'.format(video_name, obj1, obj2))
		
			if (video_name, obj1) not in self.object_features:
				with open(obj1_feature_path, 'rb') as pickle_file:
					self.object_features[(video_name, obj1)] = pickle.load(pickle_file)

			if (video_name, obj2) not in self.object_features:
				with open(obj2_feature_path, 'rb') as pickle_file:
					self.object_features[(video_name, obj2)] = pickle.load(pickle_file)

			if (video_name, obj1) not in self.object_vgg_features:
				with open(obj1_vgg_feature_path, 'rb') as pickle_file:
					self.object_vgg_features[(video_name, obj1)] = pickle.load(pickle_file)

			if (video_name, obj2) not in self.object_vgg_features:
				with open(obj2_vgg_feature_path, 'rb') as pickle_file:
					self.object_vgg_features[(video_name, obj2)] = pickle.load(pickle_file)
					
			if (video_name, obj1, obj2) not in self.pair_features:
				with open(pair_feature_path, "rb") as pickle_file:
					self.pair_features[(video_name, obj1, obj2)] = pickle.load(pickle_file)
	
			if (video_name, obj1, obj2) not in self.pair_features_blackened:
				with open(pair_feature_blackened_path, "rb") as pickle_file:
					self.pair_features_blackened[(video_name, obj1, obj2)] = pickle.load(pickle_file)


		self.word_list = []
		with open(os.path.join(data_dir, 'dict.txt'), 'r') as dict_file:
			for line in dict_file:
				line = line.strip()
				if line == '':
					continue
				self.word_list.append(line)
		
		word2vec = {}
		with open(os.path.join(data_dir, 'glove.6B.50d.txt'), 'r') as word2vec_file:
			for line in word2vec_file:
				line = line.strip()
				if line == '':
					continue
				word, *vector = line.split()
				vector = np.asarray(list(map(float, vector))).astype(np.float32)
				word2vec[word] = vector
		self.word_embed['<nil>'] = np.zeros((50, )).astype(np.float32)
		self.word_embed['<unk>'] = np.ones((50, )).astype(np.float32)
		self.word_embed['<start>'] = np.zeros((50, )).astype(np.float32)
		self.word_embed['<end>'] = np.zeros((50, )).astype(np.float32)
		
		self.word_embed['<start>'][0] = 1
		self.word_embed['<end>'][1] = 1
		
		
		self.word2idx['<nil>'] = 0
		self.word2idx['<unk>'] = 1
		self.word2idx['<start>'] = 2
		self.word2idx['<end>'] = 3
		
		for word_idx, word in enumerate(self.word_list):
			if word not in word2vec:
				raise NameError("Dictionary word {:s} is not in word2vec".format(word))
			self.word2idx[word] = word_idx+4
			self.word_embed[word] = word2vec[word]
		
		self.word_list = ['<nil>', '<unk>', '<start>', '<end>'] + self.word_list
		
	
	def __len__(self):
		return len(self.refexp_list)
	
	
	def __getitem__(self, index):
		video_name, obj1, obj2, refexp = self.refexp_list[index]
		sorted_obj1, sorted_obj2 = sorted([obj1, obj2])
		
		obj1_feature = self.object_features[(video_name, obj1)]
		obj2_feature = self.object_features[(video_name, obj2)]
		obj1_vgg_feature = self.object_vgg_features[(video_name, obj1)]
		obj2_vgg_feature = self.object_vgg_features[(video_name, obj2)]
		pair_feature = self.pair_features[(video_name, sorted_obj1, sorted_obj2)]
		pair_feature_blackened = self.pair_features_blackened[(video_name, sorted_obj1, sorted_obj2)]
		
		decoder_input = np.zeros((self.max_refexp_len+1, 50))
		decoder_output = np.zeros((self.max_refexp_len+1, )).astype(int)
		
		exps = refexp.split()
		exps = ["<start>"] + exps + ["<end>"]
		exps_embed = []
		for exp_idx, exp in enumerate(exps):
			if exp not in self.word_embed:
				exp = "<unk>"
				exps[exp_idx] = "<unk>"
			exps_embed.append(self.word_embed[exp])
		
		t_decoder = 0
		while t_decoder < len(exps_embed)-1:
			decoder_input[t_decoder] = exps_embed[t_decoder]
			decoder_output[t_decoder] = self.word2idx[exps[t_decoder+1]]
			t_decoder += 1
		
		sample = {"obj1_feature":obj1_feature,
				  "obj2_feature":obj2_feature,
				  "obj1_vgg_feature":obj1_vgg_feature,
				  "obj2_vgg_feature":obj2_vgg_feature,
				  "pair_feature":pair_feature,
				  "pair_feature_blackened":pair_feature_blackened,
				  "decoder_input":decoder_input,
				  "decoder_output":decoder_output,
				  "video_name":video_name,
				  "obj1":obj1,
				  "obj2":obj2,
				  "refexp":refexp}
		return sample

