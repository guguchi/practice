from __future__ import division
from __future__ import print_function
from six.moves import xrange
import chainer, argparse, os, sys, cupy, collections, six, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from chainer import optimizers, iterators, cuda, Variable, initializers
from chainer import links as L
from chainer import functions as F
from selu import selu, dropout_selu

class DeepModel(chainer.Chain):
	def __init__(self, num_layers=8, hidden_units=1000):
		super(DeepModel, self).__init__(
			logits=L.Linear(None, 10),
		)
		self.num_layers = num_layers
		self.activations = []
		input_units = [768] + [hidden_units] * num_layers
		for idx in xrange(num_layers):
			self.add_link("layer_%s" % idx, L.Linear(None, 1000, initialW=initializers.Normal(math.sqrt(1. / input_units[idx]))))

class SELUDeepModel(DeepModel):
	name = "SELU"
	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			out = selu(layer(out))
			if chainer.config.train == False:
				self.activations.append(xp.copy(out.data))
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class SELUAlphaDropoutDeepModel(DeepModel):
	name = "SELU+AlphaDropout"
	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			out = selu(layer(out))
			if chainer.config.train == False:
				self.activations.append(xp.copy(out.data))
			out = dropout_selu(out, ratio=0.1)
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class ReLUBatchnormDeepModel(DeepModel):
	name = "ReLU+BatchNorm"
	def __init__(self, num_layers=8):
		super(ReLUBatchnormDeepModel, self).__init__(num_layers=num_layers)
		for idx in xrange(num_layers):
			self.add_link("bn_%s" % idx, L.BatchNormalization(1000))

	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			batchnorm = getattr(self, "bn_%s" % idx)
			out = batchnorm(F.relu(layer(out)))
			if chainer.config.train == False:
				self.activations.append(xp.copy(out.data))
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class RELUDeepModel(DeepModel):
	name = "ReLU"
	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			out = F.relu(layer(out))
			if chainer.config.train == False:
				self.activations.append(xp.copy(out.data))
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

class ELUDeepModel(DeepModel):
	name = "ELU"
	def __call__(self, x, apply_softmax=True):
		del self.activations[:]
		xp = self.xp
		out = x
		for idx in xrange(self.num_layers):
			layer = getattr(self, "layer_%s" % idx)
			out = F.elu(layer(out))
			if chainer.config.train == False:
				self.activations.append(xp.copy(out.data))
		out = self.logits(out)
		if apply_softmax:
			out = F.softmax(out)
		return out

def get_mnist():
	mnist_train, mnist_test = chainer.datasets.get_mnist()
	train_data, train_label = [], []
	test_data, test_label = [], []
	for data in mnist_train:
		train_data.append(data[0])
		train_label.append(data[1])
	for data in mnist_test:
		test_data.append(data[0])
		test_label.append(data[1])
	train_data = np.asanyarray(train_data, dtype=np.float32)
	test_data = np.asanyarray(test_data, dtype=np.float32)
	train_data = (train_data - np.mean(train_data)) / np.std(train_data)
	test_data = (test_data - np.mean(test_data)) / np.std(test_data)
	return (train_data, np.asanyarray(train_label, dtype=np.int32)), (test_data, np.asanyarray(test_label, dtype=np.int32))

def compute_classification_accuracy(model, x, t):
	xp = model.xp
	batches = xp.split(x, len(x) // 100)
	scores = None
	for batch in batches:
		p = F.softmax(model(batch, apply_softmax=False)).data
		scores = p if scores is None else xp.concatenate((scores, p), axis=0)
	return float(F.accuracy(scores, Variable(t)).data)

def plot_activations(model, x, out_dir):
	try:
		os.mkdir(out_dir)
	except:
		pass

	if isinstance(model, DeepModel):
		sns.set(font_scale=0.5)
		fig = plt.figure()
		num_layers = model.num_layers

		with chainer.using_config("train", False):
			xp = model.xp
			batches = xp.split(x, len(x) // 200)
			num_layers = model.num_layers
			layer_activations = [None] * num_layers
			for batch_idx, batch in enumerate(batches):
				sys.stdout.write("\rplotting {}/{}".format(batch_idx + 1, len(batches)))
				sys.stdout.flush()
				logits = model(batch)
				for layer_idx, activations in enumerate(model.activations):
					data = cuda.to_cpu(activations).reshape((-1,))
					# append
					pool = layer_activations[layer_idx]
					pool = data if pool is None else np.concatenate((pool, data))
					layer_activations[layer_idx] = pool

			sys.stdout.write("\r")
			sys.stdout.flush()

			fig, axes = plt.subplots(1, num_layers)
			for layer_idx, (activations, ax) in enumerate(zip(layer_activations, axes)):
				ax.hist(activations, bins=100)
				# ax.set_xlim([-5, 5])
				ax.set_ylim([0, 1e6])
				ax.get_yaxis().set_major_formatter(mtick.FormatStrFormatter("%.e"))
				mean, var = xp.mean(activations), xp.var(activations)
				ax.set_title("Layer #{}\nmean: {:.3f}\nvar: {:.3f}".format(layer_idx + 1, mean, var))
				print("layer #{} - mean: {:.4f} - var: {:.4f}".format(layer_idx + 1, mean, var))

			fig.suptitle("%s Activation Distribution" % model.__class__.name)
			plt.savefig(os.path.join(out_dir, "activation.png"), dpi=350)

def train(args):
	mnist_train, mnist_test = get_mnist()

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# init model
	model = None
	if args.model.lower() == "selu":
		model = SELUDeepModel()
	elif args.model.lower() == "selu_dropout":
		model = SELUAlphaDropoutDeepModel()
	elif args.model.lower() == "relu":
		model = RELUDeepModel()
	elif args.model.lower() == "relu_bn":
		model = ReLUBatchnormDeepModel()
	elif args.model.lower() == "elu":
		model = ELUDeepModel()
	assert model is not None
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		model.to_gpu()
	xp = model.xp

	# init optimizer
	optimizer = optimizers.Adam(alpha=args.learning_rate, beta1=0.9)
	optimizer.setup(model)

	train_data, train_label = mnist_train
	test_data, test_label = mnist_test
	if args.gpu_device >= 0:
		train_data = cuda.to_gpu(train_data)
		train_label = cuda.to_gpu(train_label)
		test_data = cuda.to_gpu(test_data)
		test_label = cuda.to_gpu(test_label)
	train_loop = len(train_data) // args.batchsize
	train_indices = np.arange(len(train_data))

	# training cycle
	for epoch in xrange(1, args.epoch):
		np.random.shuffle(train_indices)	# shuffle data
		sum_loss = 0

		with chainer.using_config("train", True):
			# loop over all batches
			for itr in xrange(1, train_loop + 1):
				# sample minibatch
				batch_range = np.arange(itr * args.batchsize, min((itr + 1) * args.batchsize, len(train_data)))
				x = train_data[train_indices[batch_range]]
				t = train_label[train_indices[batch_range]]

				# to gpu
				if model.xp is cuda.cupy:
					x = cuda.to_gpu(x)
					t = cuda.to_gpu(t)

				logits = model(x, apply_softmax=False)
				loss = F.softmax_cross_entropy(logits, Variable(t))

				# update weights
				optimizer.update(lossfun=lambda: loss)

				if itr % 50 == 0 or itr == train_loop:
					sys.stdout.write("\riteration {}/{}".format(itr, train_loop))
					sys.stdout.flush()
				sum_loss += float(loss.data)

		with chainer.using_config("train", False):
			accuracy_train = compute_classification_accuracy(model, train_data, train_label)
			accuracy_test = compute_classification_accuracy(model, test_data, test_label)

		sys.stdout.write("\r\033[2KEpoch {} - loss: {:.8f} - acc: {:.5f} (train), {:.5f} (test)\n".format(epoch, sum_loss / train_loop, accuracy_train, accuracy_test))
		sys.stdout.flush()

	# plot activations
	plot_activations(model, test_data, args.model)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", "-m", type=str, default="selu")
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.002)
	parser.add_argument("--epoch", "-e", type=int, default=50)
	parser.add_argument("--batchsize", "-b", type=int, default=256)
	parser.add_argument("--seed", "-seed", type=int, default=0)
	args = parser.parse_args()
	train(args)
