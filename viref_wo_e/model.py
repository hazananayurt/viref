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
		
		self.fc1 = torch.nn.Linear(input_size, hidden_size*num_layers*2)
		self.fc2 = torch.nn.Linear(hidden_size*num_layers*2, hidden_size*num_layers)
		self.fc3 = torch.nn.Linear(hidden_size*num_layers, hidden_size*num_layers)
		
		self.bn1 = torch.nn.BatchNorm1d(hidden_size*num_layers*2)
		self.bn2 = torch.nn.BatchNorm1d(hidden_size*num_layers)
		
		self.dropout1 = torch.nn.Dropout(dropout)
		self.dropout2 = torch.nn.Dropout(dropout)		
		
		
	def forward(self, inp):
		out = self.fc1(inp)
		out = F.relu(out)
		out = self.bn1(out)
		out = self.dropout1(out)
		out = self.fc2(out)
		out = F.relu(out)
		out = self.bn2(out)
		out = self.dropout2(out)
		out = self.fc3(out)
		out = out.view(-1, self.num_layers, self.hidden_size)
		out = out.permute(1, 0, 2)
		return out


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

