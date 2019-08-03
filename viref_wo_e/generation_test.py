import os
import torch
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from .model import Encoder, Decoder
from .dataset import VirefDataset


def generation_test(args, save_path):
	use_cuda = args.cuda
	max_refexp_len = args.max_refexp_len
	batch_size = args.batch_size
	hidden_size = args.hidden_size
	dropout = args.dropout
	num_layers = args.num_layers
	dataset_split_dir = args.dataset_split_dir
	
	results_path = os.path.join('viref_wo_e', 'generation_results.txt')
	if os.path.exists(results_path):
		return
	
	beam_size = 3
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
	
	encoder.eval()
	decoder.eval()
	test_dataset = VirefDataset(args, os.path.join(dataset_split_dir, 'test_refexp.csv'), max_refexp_len=max_refexp_len)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
	
	initial_decoder_input = torch.from_numpy(test_dataset.word_embed['<start>']).unsqueeze(0).unsqueeze(0).to(device)
	
	results = {}

	word_embed = torch.cat([torch.from_numpy(test_dataset.word_embed[word]).unsqueeze(0).to(device) for word in test_dataset.word_list], dim=0)

	with torch.no_grad():
		for batch_idx, sampled_batch in enumerate(test_loader):
			print(batch_idx, '/', len(test_loader))
			obj1_feature = sampled_batch['obj1_feature'].to(device).float()
			obj2_feature = sampled_batch['obj2_feature'].to(device).float()
			obj1_vgg_feature = sampled_batch['obj1_vgg_feature'].to(device).float()
			obj2_vgg_feature = sampled_batch['obj2_vgg_feature'].to(device).float()
			pair_feature = sampled_batch['pair_feature'].to(device).float()
			pair_feature_blackened = sampled_batch['pair_feature_blackened'].to(device).float()
			
			features = [obj1_feature,
						obj2_feature,
						obj1_vgg_feature,
						obj2_vgg_feature,
						pair_feature,
						pair_feature_blackened]
			
			
			decoder_input = sampled_batch['decoder_input'].to(device).float()
			decoder_output = sampled_batch['decoder_output'].to(device)

			current_batch_size = decoder_input.shape[0]

			expanded_c0 = c0.expand(-1, current_batch_size, -1).contiguous()
			
			encoder_input = torch.cat(features, dim=1)

			h0 = encoder(encoder_input).contiguous()
			
			hn = h0
			cn = expanded_c0
			top_probabilities = [[torch.zeros(current_batch_size).to(device), initial_decoder_input.expand(current_batch_size, -1, -1).contiguous(),
								hn, cn, torch.ones(current_batch_size).int().to(device), [[] for j in range(current_batch_size)]] for i in range(beam_size)]
			
			for i in range(decoder_input.shape[1]):
				new_top_probabilities = []
				for log_probs, decoder_input_i, hi, ci, end_mask, sampled_words in top_probabilities:
					out_i, (hi, ci) = decoder(decoder_input_i, hi, ci)
					out_i = out_i.squeeze(1)
					for j in range(beam_size):
						maximum_idcs = torch.argmax(out_i, dim=1)
						step_log_probs = out_i[torch.arange(current_batch_size).to(device), maximum_idcs]
						new_end_mask = end_mask * (1-((maximum_idcs==0).int()+(maximum_idcs==3).int()))
						new_log_probs = new_end_mask.float()*step_log_probs + log_probs
						batch_results = [[] for k in range(current_batch_size)]
						new_sampled_words = []
						for k in range(maximum_idcs.size(0)):
							new_sampled_words.append(sampled_words[k]+[test_dataset.word_list[maximum_idcs[k].item()]])
						new_decoder_input_i = word_embed[maximum_idcs].unsqueeze(1)
						new_top_probabilities.append([new_log_probs, new_decoder_input_i, hi, ci, new_end_mask, new_sampled_words])
						out_i[torch.arange(current_batch_size).to(device), maximum_idcs] = float('-inf')
				top_probabilities = [[torch.zeros_like(item).to(device) for item in new_top_probabilities[0][:-1]] + [[]] for j in range(beam_size)]
				
				for j in range(current_batch_size):
					top_scorers = []
					for k in range(len(new_top_probabilities)):
						log_prob = float(new_top_probabilities[k][0][j].item())
						top_scorers.append((log_prob, new_top_probabilities[k]))
					top_scorers = sorted(top_scorers, key=(lambda x: x[0]), reverse=True)

					for k in range(beam_size):
						top_probabilities[k][0][j] = top_scorers[k][1][0][j] #log_probs
						top_probabilities[k][1][j] = top_scorers[k][1][1][j] #decoder_input_i
						top_probabilities[k][2][:, j, :] = top_scorers[k][1][2][:, j, :] #hi
						top_probabilities[k][3][:, j, :] = top_scorers[k][1][3][:, j, :] #ci
						top_probabilities[k][4][j] = top_scorers[k][1][4][j] #end_mask
						top_probabilities[k][5].append(top_scorers[k][1][5][j]) #sampled_words

			for i in range(current_batch_size):
				result_refexp = []
				for word in top_probabilities[0][-1][i]:
					if word in ['<nil>', '<end>']:
						break
					result_refexp.append(word)
				video_name = sampled_batch['video_name'][i]
				obj1 = sampled_batch['obj1'][i].item()
				obj2 = sampled_batch['obj2'][i].item()
				gt_refexp = sampled_batch['refexp'][i].split()
				key = (video_name, obj1, obj2)
				if key not in results:
					results[key] = [result_refexp, [gt_refexp]]
				else:
					results[key][1].append(gt_refexp)
	with open(results_path, 'w') as f:
		f.write(str(results))

