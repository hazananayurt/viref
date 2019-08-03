import os
import torch
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from .model import Encoder, Decoder
from .dataset import VirefDataset


def comprehension_test(args, save_path):
	use_cuda = args.cuda
	max_refexp_len = args.max_refexp_len
	batch_size = args.batch_size
	hidden_size = args.hidden_size
	dropout = args.dropout
	num_layers = args.num_layers
	dataset_split_dir = args.dataset_split_dir
	
	encoder_input_size, decoder_input_size, decoder_output_size = 6*4096, 50, 1024
	
	device = torch.device('cuda' if use_cuda else 'cpu')
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	
	encoder = Encoder(encoder_input_size, hidden_size, num_layers, dropout)
	decoder = Decoder(decoder_input_size, hidden_size, decoder_output_size, num_layers, dropout)
	
	checkpoint = torch.load(save_path)
	encoder.load_state_dict(checkpoint['encoder_state_dict'])
	decoder.load_state_dict(checkpoint['decoder_state_dict'])
	
	encoder.to(device)
	decoder.to(device)
	
	c0 = torch.zeros(num_layers, 1, hidden_size).to(device)
	
	results_folder_path = os.path.join('viref_wo_e', 'comprehension_results')
	if not os.path.exists(results_folder_path):
		os.makedirs(results_folder_path)
	
	encoder.eval()
	decoder.eval()
	
	video_names = set([])
	with open(os.path.join(dataset_split_dir, 'test_refexp.csv'), 'r') as f:
		for line in f:
			video_name, *_ = line.strip().split(',')
			video_names.add(video_name)
			
	video_names = sorted(list(video_names))

	for video_idx, video_name in enumerate(video_names):
		print(video_name, "({:d}/{:d})".format(video_idx, len(video_names)))
		path = os.path.join(results_folder_path, video_name)
		if os.path.exists(path):
			continue
		test_dataset = VirefDataset(args, os.path.join(dataset_split_dir, 'test_refexp.csv'), max_refexp_len=max_refexp_len, video_name_restriction=video_name)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
		initial_decoder_input = torch.from_numpy(test_dataset.word_embed['<start>']).unsqueeze(0).unsqueeze(0).to(device)
		video_results = []
		with torch.no_grad():
			for query_idx, query in enumerate(test_dataset):
				print('\tquery {:d}/{:d}'.format(query_idx, len(test_dataset)))
				explored_set = set([])
				decoder_input = torch.from_numpy(query['decoder_input']).unsqueeze(0).to(device).float()
				decoder_output = torch.from_numpy(query['decoder_output']).to(device)
				query_key = (query['video_name'], query['obj1'], query['obj2'])
				query_results = []
				for example_idx, example in enumerate(test_dataset):
					video_name = example['video_name']
					obj1 = example['obj1']
					obj2 = example['obj2']
					key = (video_name, obj1, obj2)
					if key in explored_set:
						continue
					explored_set.add(key)
					
					obj1_feature = torch.from_numpy(example['obj1_feature']).unsqueeze(0).to(device).float()
					obj2_feature = torch.from_numpy(example['obj2_feature']).unsqueeze(0).to(device).float()
					obj1_vgg_feature = torch.from_numpy(example['obj1_vgg_feature']).unsqueeze(0).to(device).float()
					obj2_vgg_feature = torch.from_numpy(example['obj2_vgg_feature']).unsqueeze(0).to(device).float()
					pair_feature = torch.from_numpy(example['pair_feature']).unsqueeze(0).to(device).float()
					pair_feature_blackened = torch.from_numpy(example['pair_feature_blackened']).unsqueeze(0).to(device).float()
					
					features = [obj1_feature,
								obj2_feature,
								obj1_vgg_feature,
								obj2_vgg_feature,
								pair_feature,
								pair_feature_blackened]
								
					encoder_input = torch.cat(features, dim=1)

					current_batch_size = decoder_input.shape[0]
					expanded_c0 = c0.expand(-1, current_batch_size, -1).contiguous()

					h0 = encoder(encoder_input).contiguous()
					out, _ = decoder(decoder_input, h0, expanded_c0)
					out = out.squeeze(0)

					out[torch.sum((decoder_output!=0).int()).item()-1:] = 0
					log_prob = torch.sum(out[torch.arange(decoder_output.size(0)), decoder_output]).item()

					query_results.append((log_prob, key))
				video_results.append((query_key, sorted(query_results)[::-1]))
		with open(path, 'w') as f:
			f.write(str(video_results))

