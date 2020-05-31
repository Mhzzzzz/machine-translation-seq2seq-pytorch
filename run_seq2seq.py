import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary

import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom
from nltk.translate.bleu_score import sentence_bleu

import data
import seq2seq

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
	# ======================
	# 超参数
	# ======================
	CELL = "gru"                   # rnn, gru, lstm
	BATCH_SIZE = 64
	ENC_EMBED_SIZE = 128
	DEC_EMBED_SIZE = 128
	HIDDEN_DIM = 128
	NUM_LAYERS = 2
	DROPOUT_RATE = 0.0
	EPOCH = 200
	LEARNING_RATE = 0.01
	MAX_GENERATE_LENGTH = 20
	SAVE_EVERY = 5
	ATTENTION = True

	all_var = locals()
	print()
	for var in all_var:
		if var != "var_name":
			print("{0:15}   ".format(var), all_var[var])
	print()

	# ======================
	# 数据
	# ======================
	DOMTree = xml.dom.minidom.parse('en-hr.tmx')
	collection = DOMTree.documentElement
	raw = list(collection.getElementsByTagName('tu'))
	raw_en = [raw[i].childNodes[1].childNodes[0].childNodes[0].data for i in range(len(raw))]
	raw_hr = [raw[i].childNodes[3].childNodes[0].childNodes[0].data for i in range(len(raw))]
	data_helper_en = data.DataHelper([raw_en])
	data_helper_hr = data.DataHelper([raw_hr])
	corpus = [data_helper_en.corpus, data_helper_hr.corpus]
	data_generator = data.DataGenerator(corpus)

	# ======================
	# 构建模型
	# ======================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = seq2seq.Seq2seq(
		cell=CELL,
		enc_vocab_size=data_helper_en.vocab_size,
		enc_embed_size=ENC_EMBED_SIZE,
		enc_hidden_dim=HIDDEN_DIM,
		num_layers=NUM_LAYERS,
		dec_vocab_size=data_helper_hr.vocab_size,
		dec_embed_size=DEC_EMBED_SIZE,
		dropout_rate=DROPOUT_RATE,
		use_attention=ATTENTION
	)
	model.to(device)
	summary(model, [(20,), (20,)])
	criteration = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	print()

	# ======================
	# 训练与测试
	# ======================
	for epoch in range(EPOCH):
		generator_train = data_generator.train_generator(BATCH_SIZE)
		generator_test = data_generator.test_generator(BATCH_SIZE)
		train_loss = []
		while True:
			try:
				text = generator_train.__next__()
			except:
				break
			text = text[0]
			optimizer.zero_grad()
			x_enc = torch.from_numpy(text[0]).to(device)
			x_dec = torch.from_numpy(text[1][:, :-1]).to(device)
			y = model([x_enc, x_dec])
			loss = criteration(y.reshape(-1, data_helper_hr.vocab_size),
			                   torch.from_numpy(text[1][:, 1:]).reshape(-1).long().to(device))
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())

		test_loss = []
		while True:
			with torch.no_grad():
				try:
					text = generator_test.__next__()
				except:
					break
				text = text[0]
				x_enc = torch.from_numpy(text[0]).to(device)
				x_dec = torch.from_numpy(text[1][:, :-1]).to(device)
				y = model([x_enc, x_dec])
				loss = criteration(y.reshape(-1, data_helper_hr.vocab_size),
				                   torch.from_numpy(text[1][:, 1:]).reshape(-1).long().to(device))
				test_loss.append(loss.item())

		print('epoch {:d}   training loss {:.4f}    test loss {:.4f}'
		      .format(epoch + 1, np.mean(train_loss), np.mean(test_loss)))

		if (epoch + 1) % SAVE_EVERY == 0:
			print('-----------------------------------------------------')
			print('saving parameters')
			os.makedirs('models', exist_ok=True)
			torch.save(model.state_dict(), 'models/seq2seq-' + str(epoch) + '.pkl')

			with torch.no_grad():
				# 生成文本
				generator_test = data_generator.test_generator(3)
				text = generator_test.__next__()
				text = text[0]
				x = [torch.from_numpy(text[0]).to(device),
				     torch.LongTensor([[data_helper_hr.w2i['_BOS']]] * 3).to(device)]
				for i in range(MAX_GENERATE_LENGTH):
					samp = model.sample(x)
					x[1] = torch.cat([x[1], samp], dim=1)
				x[1] = x[1].cpu().numpy()
			for i in range(x[0].shape[0]):
				print(' '.join([data_helper_en.i2w[_] for _ in list(text[0][i, :]) if _ not in
				                [data_helper_en.w2i['_BOS'], data_helper_en.w2i['_EOS'], data_helper_en.w2i['_PAD']]]))
				print(' '.join([data_helper_hr.i2w[_] for _ in list(text[1][i, :]) if _ not in
				                [data_helper_hr.w2i['_BOS'], data_helper_hr.w2i['_EOS'], data_helper_hr.w2i['_PAD']]]))
				print(' '.join([data_helper_hr.i2w[_] for _ in list(x[1][i, :]) if _ not in
				                [data_helper_hr.w2i['_BOS'], data_helper_hr.w2i['_EOS'], data_helper_hr.w2i['_PAD']]]))
				print()
			print('-----------------------------------------------------')


if __name__ == '__main__':
	main()
