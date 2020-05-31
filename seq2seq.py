import torch
import torch.nn as nn


class Seq2seq(nn.Module):
	def __init__(self,
	             cell,
	             enc_vocab_size,
	             enc_embed_size,
	             enc_hidden_dim,
	             num_layers,
	             dec_vocab_size,
	             dec_embed_size,
	             dropout_rate,
	             use_attention=True):
		super(Seq2seq, self).__init__()
		self._cell = cell
		self._use_attention = use_attention
		dec_input_size = dec_embed_size
		if self._use_attention:
			dec_input_size += 2 * enc_hidden_dim

		self.enc_embedding = nn.Embedding(enc_vocab_size, enc_embed_size)
		self.dec_embedding = nn.Embedding(dec_vocab_size, dec_embed_size)
		if cell == 'rnn':
			self.enc_rnn = nn.RNN(enc_embed_size, enc_hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			self.dec_rnn = nn.RNN(dec_input_size, 2 * enc_hidden_dim, num_layers, dropout=dropout_rate)
		elif cell == 'gru':
			self.enc_rnn = nn.GRU(enc_embed_size, enc_hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			self.dec_rnn = nn.GRU(dec_input_size, 2 * enc_hidden_dim, num_layers, dropout=dropout_rate)
		elif cell == 'lstm':
			self.enc_rnn = nn.LSTM(enc_embed_size, enc_hidden_dim, num_layers, dropout=dropout_rate, bidirectional=True)
			self.dec_rnn = nn.LSTM(dec_input_size, 2 * enc_hidden_dim, num_layers, dropout=dropout_rate)
		else:
			raise Exception("no such rnn cell")

		self.output_layer = nn.Linear(2 * enc_hidden_dim, dec_vocab_size)
		self.log_softmax = nn.LogSoftmax(dim=2)

	def forward(self, x):
		"""
		:param x: [(N,L), (N,L)]
		:return:
		"""
		x_enc = x[0]
		x_dec = x[1]
		x_enc = x_enc.long()
		x_enc = self.enc_embedding(x_enc)                   # (N,L,embed)
		x_enc = x_enc.permute(1, 0, 2)                      # (L,N,embed)
		h_all_enc, h_out_enc = self.enc_rnn(x_enc)          # (L,N,2*h)    (2*layer,N,h)
		if self._cell == "lstm":
			h, c = h_out_enc
			h = h.permute(1, 0, 2)
			h = h.reshape(h.shape[0], -1, 2 * h.shape[-1])
			h = h.permute(1, 0, 2)
			h = h.contiguous()
			c = c.permute(1, 0, 2)
			c = c.reshape(c.shape[0], -1, 2 * c.shape[-1])
			c = c.permute(1, 0, 2)
			c = c.contiguous()
			h_out_enc = (h, c)
		else:
			h_out_enc = h_out_enc.permute(1, 0, 2)                                              # (N,2*layer,h)
			h_out_enc = h_out_enc.reshape(h_out_enc.shape[0], -1, 2 * h_out_enc.shape[-1])      # (N,layer,2*h)
			h_out_enc = h_out_enc.permute(1, 0, 2)                                              # (layer,N,2*h)
			h_out_enc = h_out_enc.contiguous()

		x_dec = x_dec.long()
		x_dec = self.dec_embedding(x_dec)       # (N,L,embed)
		x_dec = x_dec.permute(1, 0, 2)          # (L,N,embed)

		if self._use_attention:
			h_all_dec = []
			h_last = h_out_enc
			for i in range(x_dec.shape[0]):
				# attention
				if self._cell == 'lstm':
					e = torch.mul(h_all_enc, h_last[0][-1].unsqueeze(dim=0))
				else:
					e = torch.mul(h_all_enc, h_last[0][-1].unsqueeze(dim=0))    # (L,N,2*h) = (L,N,2*h)·(1,N,2*h)
				e = torch.sum(e, dim=2)                                         # (L,N)
				alpha = torch.softmax(e, dim=0)
				c = torch.mul(alpha.unsqueeze(2), h_all_enc)                    # (L,N,2*h) = (L,N,1)·(L,N,2*h)
				c = torch.sum(c, dim=0)                                         # (N,2*h)
				# concat
				input = torch.cat([x_dec[i], c], dim=1).unsqueeze(dim=0)        # (1,N,embed+2*h)
				# step
				h_i_all, h_last = self.dec_rnn(input, h_last)                   # (1,N,2*h) (layer,N,2*h)
				h_all_dec.append(h_i_all)
			h_all_dec = torch.cat(h_all_dec, dim=0)

		else:
			h_all_dec, __ = self.dec_rnn(x_dec, h_out_enc)      # (L,N,2*h)
		h_all_dec = h_all_dec.permute(1, 0, 2)
		_ = self.output_layer(h_all_dec)
		_ = self.log_softmax(_)
		return _

	def sample(self, x):
		log_prob = self.forward(x)
		prob = torch.exp(log_prob)[:, -1, :]
		return torch.multinomial(prob, 1)
