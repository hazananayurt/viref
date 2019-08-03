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
		
		
	def forward(self, inp, h0, c0):
		out, (hn, cn) = self.lstm(inp, (h0, c0))
		return out, (hn, cn)
		
		
	def init_h_c(self):
		h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
		c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
		return h0, c0


class Decoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
		super(Decoder, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
		
		self.lstm = torch.nn.LSTM(input_size=input_size,
								  hidden_size=hidden_size,
								  num_layers=num_layers,
								  batch_first=True,
								  dropout=dropout)
		self.fc = torch.nn.Linear(hidden_size, output_size)
		
		
	def forward(self, inp, h0, c0):
		out, (hn, cn) = self.lstm(inp, (h0, c0))
		batch_size = inp.size(0)
		out = out.contiguous()
		out = out.view(-1, out.size(2))
		out = self.fc(out)
		out = out.view(batch_size, -1, out.size(1))
		out = F.log_softmax(out, dim=2)
		return out, (hn, cn)

