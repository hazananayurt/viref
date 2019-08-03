import os
import torch
import math
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from .model import Encoder, Decoder
from .dataset import VirefDataset



def train(args, save_path=False):
	use_cuda = args.cuda
	max_refexp_len = args.max_refexp_len
	batch_size = args.batch_size
	epoch = args.epoch
	hidden_size = args.hidden_size
	dropout = args.dropout
	learning_rate = args.learning_rate
	num_layers = args.num_layers
	dataset_split_dir = args.dataset_split_dir
	
	encoder_input_size, decoder_input_size, decoder_output_size = 6*4096, 50, 1024
	
	device = torch.device('cuda' if use_cuda else 'cpu')
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	
	train_dataset = VirefDataset(args, os.path.join(dataset_split_dir, 'train_refexp.csv'), max_refexp_len=max_refexp_len)
	val_dataset = VirefDataset(args, os.path.join(dataset_split_dir, 'val_refexp.csv'), max_refexp_len=max_refexp_len)
	
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

	encoder = Encoder(encoder_input_size, hidden_size, num_layers, dropout)
	decoder = Decoder(decoder_input_size, hidden_size, decoder_output_size, num_layers, dropout)
	
	encoder = encoder.to(device)
	decoder = decoder.to(device)
	
	c0 = torch.zeros(num_layers, 1, hidden_size).to(device)
	
	optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=learning_rate)
	
	best_val_loss = float('inf')
	
	for epoch_idx in range(epoch):
		print('epoch {:d}'.format(epoch_idx))
		for batch_idx, sampled_batch in enumerate(train_loader):
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
			
			optimizer.zero_grad()
			
			expanded_c0 = c0.expand(-1, current_batch_size, -1).contiguous()
			
			encoder_input = torch.cat(features, dim=1)
			
			h0 = encoder(encoder_input).contiguous()
			out, _ = decoder(decoder_input, h0, expanded_c0)
			
			out = out.contiguous().view(-1, out.size(-1))

			loss = F.nll_loss(out, decoder_output.view(-1))
			
			loss.backward()
			optimizer.step()
			
			if batch_idx % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch_idx, batch_idx * batch_size, len(train_dataset),
						100. * batch_idx / len(train_loader), loss.item()))
			if batch_idx%100 == 99:
				with torch.no_grad():
					total_val_loss = 0
					num_val_examples = 0
					val_iter = 0
					for val_sampled_batch in val_loader:
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
						num_val_examples += current_batch_size
						optimizer.zero_grad()
						
						h0 = encoder(encoder_input).contiguous()
						out, _ = decoder(decoder_input, h0, expanded_c0)
						
						
						out = out.contiguous().view(-1, out.size(-1))

						loss = F.nll_loss(out, decoder_output.view(-1))
						total_val_loss += float(torch.sum(loss).item())
						val_iter += 1

					val_loss = total_val_loss / val_iter
					if val_loss < best_val_loss:
						if save_path is not False:
							torch.save({
										'encoder_state_dict': encoder.state_dict(),
										'decoder_state_dict': decoder.state_dict(),
										}, save_path)
						best_val_loss = val_loss
					print('val_loss: {:f}, best_val_loss: {:f}'.format(val_loss, best_val_loss))

