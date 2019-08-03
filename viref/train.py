import os
import torch
import math
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from .model import Model
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
	
	encoder_input_size, decoder_input_size, decoder_output_size, num_features = 5*4096, 50, 1024, 5
	
	device = torch.device('cuda' if use_cuda else 'cpu')
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	
	train_dataset = VirefDataset(args, os.path.join(dataset_split_dir, 'train_refexp.csv'), max_refexp_len=max_refexp_len)
	val_dataset = VirefDataset(args, os.path.join(dataset_split_dir, 'val_refexp.csv'), max_refexp_len=max_refexp_len)
	
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
	
	model = Model(encoder_input_size, decoder_input_size, decoder_output_size, hidden_size, num_layers, num_features, dropout)
	
	model = model.to(device)
	
	h0 = torch.zeros(num_layers, 1, hidden_size).to(device)
	c0 = torch.zeros(num_layers, 1, hidden_size).to(device)
	initial_scale_weights = F.softmax(torch.randn(1, num_features)*(math.sqrt(2.0/num_features)), dim=1).to(device)
	
	h0 = Parameter(h0)
	initial_scale_weights = Parameter(initial_scale_weights)
	optimizer = optim.Adam(list(model.parameters())+[h0, initial_scale_weights], lr=learning_rate)
	
	best_val_loss = float('inf')
	
	for epoch_idx in range(epoch):
		print('epoch {:d}'.format(epoch_idx))
		for batch_idx, sampled_batch in enumerate(train_loader):
			obj1_vgg_input = sampled_batch['obj1_vgg_input'].to(device).float()
			obj2_vgg_input = sampled_batch['obj2_vgg_input'].to(device).float()
			obj1_mask_input = sampled_batch['obj1_mask_input'].to(device).float()
			obj2_mask_input = sampled_batch['obj2_mask_input'].to(device).float()
			scene_vgg_input = sampled_batch['scene_vgg_input'].to(device).float()
			
			features = [obj1_vgg_input,
						obj2_vgg_input,
						obj1_mask_input,
						obj2_mask_input,
						scene_vgg_input]
			
			decoder_input = sampled_batch['decoder_input'].to(device).float()
			decoder_output = sampled_batch['decoder_output'].to(device)
			
			current_batch_size = decoder_input.shape[0]
			
			optimizer.zero_grad()
			
			out = model(features, initial_scale_weights.expand(current_batch_size, -1).contiguous(), decoder_input, h0.expand(-1, current_batch_size, -1).contiguous(), c0.expand(-1, current_batch_size, -1).contiguous())
			
			out = out.contiguous().view(-1, out.size(-1))

			loss = F.nll_loss(out, decoder_output.view(-1))
			
			loss.backward()
			optimizer.step()
			
			if batch_idx % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch_idx, batch_idx * batch_size, len(train_dataset),
						100. * batch_idx / len(train_loader), loss.item()))
			if batch_idx %100 == 99:
				with torch.no_grad():
					total_val_loss = 0
					num_val_examples = 0
					val_iter = 0
					for val_sampled_batch in val_loader:
						obj1_vgg_input = val_sampled_batch['obj1_vgg_input'].to(device).float()
						obj2_vgg_input = val_sampled_batch['obj2_vgg_input'].to(device).float()
						obj1_mask_input = val_sampled_batch['obj1_mask_input'].to(device).float()
						obj2_mask_input = val_sampled_batch['obj2_mask_input'].to(device).float()
						scene_vgg_input = val_sampled_batch['scene_vgg_input'].to(device).float()
						
						features = [obj1_vgg_input,
									obj2_vgg_input,
									obj1_mask_input,
									obj2_mask_input,
									scene_vgg_input]
						
						decoder_input = val_sampled_batch['decoder_input'].to(device).float()
						decoder_output = val_sampled_batch['decoder_output'].to(device)
						
						current_batch_size = decoder_input.shape[0]
						num_val_examples += current_batch_size
						optimizer.zero_grad()
						
						out = model(features, initial_scale_weights.expand(current_batch_size, -1).contiguous(), decoder_input, h0.expand(-1, current_batch_size, -1).contiguous(), c0.expand(-1, current_batch_size, -1).contiguous())
						
						out = out.contiguous().view(-1, out.size(-1))

						loss = F.nll_loss(out, decoder_output.view(-1))
						total_val_loss += float(torch.sum(loss).item())
						val_iter += 1
					val_loss = total_val_loss / val_iter
					if val_loss < best_val_loss:
						if save_path is not False:
							torch.save({
										'model_state_dict': model.state_dict(),
										'h0':h0,
										'initial_scale_weights':initial_scale_weights
										}, save_path)
						best_val_loss = val_loss
					print('val_loss: {:f}, best_val_loss: {:f}'.format(val_loss, best_val_loss))

