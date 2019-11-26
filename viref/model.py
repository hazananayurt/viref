import torch
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F

class Encoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, dropout):
		super(Encoder, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		
		self.lstm = torch.nn.LSTM(input_size=input_size,
								  hidden_size=hidden_size,
								  num_layers=num_layers,
								  batch_first=True,
								  dropout=dropout)
		
		
	def forward(self, features, scale_weights, h0, c0):
		scaled_features = []
		for feature_idx, feature in enumerate(features):
			scaled_features.append(feature*scale_weights[:, feature_idx].unsqueeze(1).unsqueeze(1))
		inp = torch.cat(scaled_features, dim=2)
		out, (hn, cn) = self.lstm(inp, (h0, c0))
		return out, (hn, cn)


class Decoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, dropout):
		super(Decoder, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		
		self.lstm = torch.nn.LSTM(input_size=input_size,
								  hidden_size=hidden_size,
								  num_layers=num_layers,
								  batch_first=True,
								  dropout=dropout)
		
		
	def forward(self, inp, h0, c0):
		out, (hn, cn) = self.lstm(inp, (h0, c0))
		return out, (hn, cn)
		
		
class FF1(torch.nn.Module):
	def __init__(self, hidden_size, num_features):
		super(FF1, self).__init__()
		
		self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
		self.fc2 = torch.nn.Linear(hidden_size, int(hidden_size/2))
		self.fc3 = torch.nn.Linear(int(hidden_size/2), num_features)
		self.softmax = torch.nn.Softmax(dim=2)
		
	def forward(self, decoder_out):
		batch_size = decoder_out.size(0)
		out = decoder_out
		out = out.contiguous()
		out = out.view(-1, out.size(2))
		out = self.fc1(out)
		out = F.relu(out)
		out = self.fc2(out)
		out = F.relu(out)
		out = self.fc3(out)
		out = out.view(batch_size, -1, out.size(1))
		out = self.softmax(out)
		return out
		
		

class FF2(torch.nn.Module):
	def __init__(self, hidden_size, num_layers, decoder_output_size):
		super(FF2, self).__init__()
		
		self.fc1 = torch.nn.Linear(hidden_size*(num_layers+1), hidden_size*2)
		self.fc2 = torch.nn.Linear(hidden_size*2, hidden_size)
		self.fc3 = torch.nn.Linear(hidden_size, decoder_output_size)
		
		
	def forward(self, decoder_hidden_state, attended_features):
		batch_size = decoder_hidden_state.size(0)
		inp2 = attended_features.permute(1, 0, 2).contiguous().view(batch_size, -1)
		inp = torch.cat([decoder_hidden_state, inp2], dim=1)
		out = self.fc1(inp)
		out = F.relu(out)
		out = self.fc2(out)
		out = F.relu(out)
		out = self.fc3(out)
		out = F.log_softmax(out, dim=1)
		return out

		
class Model(torch.nn.Module):
	def __init__(self, encoder_input_size, decoder_input_size, decoder_output_size, hidden_size, num_layers, num_features, dropout):
		super(Model, self).__init__()
		
		self.encoder_input_size = encoder_input_size
		self.decoder_input_size = decoder_input_size
		self.decoder_output_size = decoder_output_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		
		self.encoder = Encoder(encoder_input_size, hidden_size, num_layers, dropout)
		self.decoder = Decoder(decoder_input_size, hidden_size, num_layers, dropout)

		self.ff1 = FF1(hidden_size, num_features)
		self.ff2 = FF2(hidden_size, num_layers, decoder_output_size)
		
		
	def forward(self, features, initial_scale_weights, decoder_input, h0, c0):
		batch_size = features[0].shape[0]
		_, (hn, cn) = self.encoder(features, initial_scale_weights, h0, c0)
		decoder_out, _ = self.decoder(decoder_input, hn, cn)
		out = self.ff1(decoder_out)
		final_out_list = []
		for i in range(int(out.size(1))):
			scale_weights = out[:, i, :]
			#print(scale_weights)
			_, (hn, _) = self.encoder(features, scale_weights, h0, c0)
			out_i = self.ff2(decoder_out[:, i, :], hn)
			final_out_list.append(out_i.unsqueeze(0))
		out = torch.cat(final_out_list, dim=0)
		out = out.permute(1, 0, 2)
		return out

