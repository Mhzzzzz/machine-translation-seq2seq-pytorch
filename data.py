import numpy as np
import collections


class DataHelper(object):
	def __init__(self, raw, word_drop=5,  use_label=False, use_length=False):
		assert (use_label and (len(raw) == 2)) or ((not use_label) and (len(raw) == 1))
		self._word_drop = word_drop

		self.use_label = use_label
		self.use_length = use_length

		self.corpus = None
		self.corpus_num = 0
		if self.use_label:
			self.label = None
		if self.use_length:
			self.corpus_length = None
			self.max_sentence_length = 0
			self.min_sentence_length = 0

		self.vocab_size = 0
		self.vocab_size_raw = 0
		self.sentence_num = 0
		self.word_num = 0

		self.w2i = {}
		self.i2w = {}

		sentences = []
		for _ in raw:
			sentences += _

		self._build_vocabulary(sentences)
		if self.use_length:
			self.corpus, self.corpus_length = self._build_corpus(sentences)
		else:
			self.corpus = self._build_corpus(sentences)
		if self.use_label:
			self.label = self._build_label(raw)

	def _build_label(self, raw):
		label = [0]*len(raw[0]) + [1]*len(raw[1])
		return np.array(label)

	def _build_vocabulary(self, sentences):
		self.sentence_num = len(sentences)
		words = []
		for sentence in sentences:
			words += sentence.split(' ')
		self.word_num = len(words)
		word_distribution = sorted(collections.Counter(words).items(), key=lambda x: x[1], reverse=True)
		self.vocab_size_raw = len(word_distribution)
		self.w2i['_PAD'] = 0
		self.w2i['_UNK'] = 1
		self.w2i['_BOS'] = 2
		self.w2i['_EOS'] = 3
		self.i2w[0] = '_PAD'
		self.i2w[1] = '_UNK'
		self.i2w[2] = '_BOS'
		self.i2w[3] = '_EOS'
		for (word, value) in word_distribution:
			if value > self._word_drop:
				self.w2i[word] = len(self.w2i)
				self.i2w[len(self.i2w)] = word
		self.vocab_size = len(self.i2w)

	def _build_corpus(self, sentences):
		def _transfer(word):
			try:
				return self.w2i[word]
			except:
				return self.w2i['_UNK']
		corpus = [[self.w2i["_BOS"]] + list(map(_transfer, sentence.split(' '))) + [self.w2i["_EOS"]] for sentence in sentences]
		if self.use_length:
			corpus_length = np.array([len(i) for i in corpus])
			self.max_sentence_length = corpus_length.max()
			self.min_sentence_length = corpus_length.min()
			return np.array(corpus), np.array(corpus_length)
		else:
			return np.array(corpus)


class DataGenerator(object):
	def __init__(self, corpus, ratio=0.8, corpus_length=None, label=None):
		"""
		:param corpus: [ndarray(N,), ndarray(N,)]
		:param corpus_length: [ndarray(N,), ndarray(N,)]
		:param label: [ndarray(N,), ndarray(N,)]
		"""
		if corpus_length is not None:
			self.use_length = True
		else:
			self.use_length = False
		if label is not None:
			self.use_label = True
		else:
			self.use_label = False

		self.sentence_num = corpus[0].shape[0]
		self._split(corpus, ratio, corpus_length, label)

	def _split(self, corpus, ratio, corpus_length=None, label=None):
		indices = list(range(self.sentence_num))
		np.random.shuffle(indices)
		self.train = [_[indices[:int(self.sentence_num * ratio)]] for _ in corpus]
		self.train_num = len(self.train[0])
		self.test = [_[indices[int(self.sentence_num * ratio):]] for _ in corpus]
		self.test_num = len(self.test[0])
		if self.use_length:
			self.train_length = [_[indices[:int(self.sentence_num * ratio)]] for _ in corpus_length]
			self.test_length = [_[indices[int(self.sentence_num * ratio):]] for _ in corpus_length]
		if self.use_label:
			self.label_train = [_[indices[:int(self.sentence_num * ratio)]] for _ in label]
			self.label_test = [_[indices[int(self.sentence_num*ratio):]] for _ in label]

	def _padding(self, batch_data):
		max_length = max([len(i) for i in batch_data])
		for i in range(len(batch_data)):
			batch_data[i] += [0] * (max_length - len(batch_data[i]))
		return np.array(list(batch_data))

	def train_generator(self, batch_size, shuffle=True):
		indices = list(range(self.train_num))
		if shuffle:
			np.random.shuffle(indices)
		while True:
			batch_indices = indices[0:batch_size]                   # 产生一个batch的index
			indices = indices[batch_size:]                          # 去掉本次index
			if len(batch_indices) == 0:
				return True
			batch_data = [_[batch_indices] for _ in self.train]
			batch_data = [self._padding(_) for _ in batch_data]
			result = [batch_data]
			if self.use_length:
				batch_length = [_[batch_indices] for _ in self.train_length]
				result.append(batch_length)
			if self.use_label:
				batch_label = [_[batch_indices] for _ in self.label_train]
				result.append(batch_label)
			yield tuple(result)

	def test_generator(self, batch_size, shuffle=True):
		indices = list(range(self.test_num))
		if shuffle:
			np.random.shuffle(indices)
		while True:
			batch_indices = indices[0:batch_size]                   # 产生一个batch的index
			indices = indices[batch_size:]                          # 去掉本次index
			if len(batch_indices) == 0:
				return True
			batch_data = [_[batch_indices] for _ in self.test]
			batch_data = [self._padding(_) for _ in batch_data]
			result = [batch_data]
			if self.use_length:
				batch_length = [_[batch_indices] for _ in self.test_length]
				result.append(batch_length)
			if self.use_label:
				batch_label = [_[batch_indices] for _ in self.label_test]
				result.append(batch_label)
			yield tuple(result)


if __name__ == '__main__':
	"""
	with open('../_data/rt-polaritydata/rt-polarity.pos', 'r', encoding='Windows-1252') as f:
		raw_pos = list(f.readlines())
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	with open('../_data/rt-polaritydata/rt-polarity.neg', 'r', encoding='Windows-1252') as f:
		raw_neg = list(f.readlines())
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
	data_helper = DataHelper([raw_pos, raw_neg], use_length=True, use_label=True)
	generator = data_helper.train_generator(64)
	i = 0
	a, b, c = generator.__next__()
	while True:
		try:
			a, b, c = generator.__next__()
			i += 1
			print(i)
		except:
			break"""
	pass






