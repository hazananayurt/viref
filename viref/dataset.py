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
		self.object_start_end = {}
		self.video_names = set([])
		self.object_vgg_features = {}
		self.object_mask_vgg_features = {}
		self.scene_vgg_features = {}
		self.word_embed = {}
		self.word2idx = {}
		self.max_refexp_len = max_refexp_len
		self.sample_periods = {}
		self.max_encoder_input_len = 0
		self.black_frame = None
		self.video_name_restriction = video_name_restriction
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
			self.object_start_end[(video_name, obj1)] = [99999999, -1]
			self.object_start_end[(video_name, obj2)] = [99999999, -1]
			self.video_names.add(video_name)
		
		for filename in os.listdir(os.path.join(features_dir, 'object_sampled')):	
			video_name, rest = filename.split('__')
			frame_no, obj_id = list(map(int, rest.split('_')))
			if (video_name, obj_id) not in self.object_start_end:
				continue
			if video_name not in self.video_names:
				continue
			start_end = self.object_start_end[(video_name, obj_id)]
			if frame_no < start_end[0]:
				start_end[0] = frame_no
			if frame_no > start_end[1]:
				start_end[1] = frame_no	
			with open(os.path.join(features_dir, 'object_sampled', filename), 'rb') as pickle_file:
				self.object_vgg_features[(video_name, obj_id, frame_no)] = pickle.load(pickle_file).astype(np.float32)

		for filename in os.listdir(os.path.join(features_dir, 'object_mask')):
			video_name, rest = filename.split('__')
			frame_no, obj_id = list(map(int, rest.split('_')))
			if (video_name, obj_id) not in self.object_start_end:
				continue
			if video_name not in self.video_names:
				continue
			with open(os.path.join(features_dir, 'object_mask', filename), 'rb') as pickle_file:
				self.object_mask_vgg_features[(video_name, obj_id, frame_no)] = pickle.load(pickle_file).astype(np.float32)
				
		for filename in os.listdir(os.path.join(features_dir, 'scene')):
			video_name, frame_no = filename.split('__')
			frame_no = int(frame_no)
			if video_name not in self.video_names:
				continue
			with open(os.path.join(features_dir, 'scene', filename), 'rb') as pickle_file:
				self.scene_vgg_features[(video_name, frame_no)] = pickle.load(pickle_file).astype(np.float32)
				
		with open(os.path.join(features_dir, 'black_frame'), 'rb') as pickle_file:
			self.black_frame = pickle.load(pickle_file).astype(np.float32)
		
		with open(os.path.join(features_dir, 'video_fps'), newline='') as csv_file:
			reader = csv.reader(csv_file, delimiter=' ')
			for video_name, fps in reader:
				if video_name not in self.video_names:
					continue
				fps = float(fps)
				self.sample_periods[video_name] = int(fps/1)
		
		for i in range(len(self.refexp_list)):
			start, end, sample_period = self.calculate_start_end_sample_period(i)
			encoder_input_len = int((end-start)/sample_period) + 1
			if encoder_input_len > self.max_encoder_input_len:
				self.max_encoder_input_len = encoder_input_len
		if self.video_name_restriction is not None:
			self.max_encoder_input_len = 42
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
		
	def calculate_start_end_sample_period(self, index):
		video_name, obj1, obj2, refexp = self.refexp_list[index]
		sample_period = self.sample_periods[video_name]
		
		start1, end1 = self.object_start_end[(video_name, obj1)]
		start2, end2 = self.object_start_end[(video_name, obj2)]
		start = min(start1, start1)
		end = max(end1, end2)
		
		return start, end, sample_period
	
	
	def __len__(self):
		return len(self.refexp_list)
	
	
	def __getitem__(self, index):
		obj1_vgg_input = np.zeros((self.max_encoder_input_len, 4096))
		obj2_vgg_input = np.zeros((self.max_encoder_input_len, 4096))
		obj1_mask_input = np.zeros((self.max_encoder_input_len, 4096))
		obj2_mask_input = np.zeros((self.max_encoder_input_len, 4096))
		scene_vgg_input = np.zeros((self.max_encoder_input_len, 4096))
		decoder_input = np.zeros((self.max_refexp_len+1, 50))
		decoder_output = np.zeros((self.max_refexp_len+1, )).astype(int)
		
		video_name, obj1, obj2, refexp = self.refexp_list[index]
		start, end, sample_period = self.calculate_start_end_sample_period(index)
		
		frame_no = start
		t_encoder = 0
		while frame_no <= end:
			if (video_name, obj1, frame_no) in self.object_vgg_features:
				obj1_vgg = self.object_vgg_features[(video_name, obj1, frame_no)]
			else:
				obj1_vgg = self.black_frame
				
			if (video_name, obj2, frame_no) in self.object_vgg_features:
				obj2_vgg = self.object_vgg_features[(video_name, obj2, frame_no)]
			else:
				obj2_vgg = self.black_frame
			
			if (video_name, obj1, frame_no) in self.object_mask_vgg_features:
				obj1_mask = self.object_mask_vgg_features[(video_name, obj1, frame_no)]
			else:
				obj1_mask = self.black_frame
				
			if (video_name, obj2, frame_no) in self.object_mask_vgg_features:
				obj2_mask = self.object_mask_vgg_features[(video_name, obj2, frame_no)]
			else:
				obj2_mask = self.black_frame
			
			if (video_name, frame_no) in self.scene_vgg_features:
				scene_vgg = self.scene_vgg_features[(video_name, frame_no)]
			else:
				raise NameError("Scene could not be found")
			
			obj1_vgg_input[t_encoder] = obj1_vgg
			obj2_vgg_input[t_encoder] = obj2_vgg
			obj1_mask_input[t_encoder] = obj1_mask
			obj2_mask_input[t_encoder] = obj2_mask
			scene_vgg_input[t_encoder] = scene_vgg
			t_encoder += 1
			frame_no += sample_period	
		
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
		
		sample = {"obj1_vgg_input":obj1_vgg_input,
				  "obj2_vgg_input":obj2_vgg_input,
				  "obj1_mask_input":obj1_mask_input,
				  "obj2_mask_input":obj2_mask_input,
				  "scene_vgg_input":scene_vgg_input,
				  "decoder_input":decoder_input,
				  "decoder_output":decoder_output,
				  "video_name":video_name,
				  "obj1":obj1,
				  "obj2":obj2,
				  "refexp":refexp}
		return sample

