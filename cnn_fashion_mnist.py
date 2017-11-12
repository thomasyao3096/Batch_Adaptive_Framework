import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load_fashion_mnist import mnist
from load_fashion_mnist import prepare_data
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from batch_adaptive_optimizers_theano import batch_adaptive_optimizer
from collections import OrderedDict


srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return floatX(np.random.randn(*shape) * 0.01)

def init_params():
	params = OrderedDict()
	params['W'] = init_weights((32, 1, 3, 3))	# 3*3 depth = 1 32 convolution kernels
	params['W2'] = init_weights((64, 32, 3, 3))
	params['W3'] = init_weights((128, 64, 3, 3))
	params['W4'] = init_weights((128 * 3 * 3, 625))
	params['W_o'] = init_weights((625, 10))
	return params

# make the params shared variables
def init_tparams(params):
	tparams = OrderedDict()
	for kk, pp in params.items():
		tparams[kk] = theano.shared(params[kk], name=kk)
	return tparams

def load_params(path, params):
   pp = np.load(path)
   for kk, vv in params.items():
      if kk not in pp:
          raise Warning('%s is not in the archive' % kk)
      params[kk] = np.array(pp[kk], dtype = theano.config.floatX)
   return params

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, tparams, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, tparams['W'], border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, tparams['W2']))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, tparams['W3']))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, tparams['W4']))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, tparams['W_o']))
    return l1, l2, l3, l4, pyx

def main():
	trX, teX, trY, teY = mnist(onehot=True)

	trX = trX.reshape(-1, 1, 28, 28)
	teX = teX.reshape(-1, 1, 28, 28)

	train_data = (trX, trY)
	valid_data = (teX, teY)

	X = T.tensor4(dtype=theano.config.floatX)
	Y = T.matrix(dtype='int64')
	
	params = init_params()
	params = load_params('fashion_mnist_params.npz', params)
	tparams = init_tparams(params)

	noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, tparams, 0.2, 0.5) # this is for training, needs dropout
	l1, l2, l3, l4, py_x = model(X, tparams, 0., 0.) # this is for predicting, no need to dropout

	y_x = T.argmax(py_x, axis=1)

	cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y)) 
	grads = T.grad(cost, wrt = list(tparams.values()))

	f_cost = theano.function(inputs=[X, Y], outputs=cost, allow_input_downcast=True)
	f_pred = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
	f_grad = theano.function(inputs=[X, Y], outputs=grads, allow_input_downcast=True)
	# updates = RMSprop(cost, params, lr=0.001)
	AdaBatch = batch_adaptive_optimizer(
									tparams,
									train_data,
									60000,
									valid_data,
									10000,
									prepare_data,
									f_cost,
									f_grad,
									f_pred,
									'fashion_mnist',
									lrate = 0.02,
									gamma = 0.9,
									max_epoch = 100,
									max_update = 200000,
									m0 = 32
									)
	AdaBatch.train('MSGD')


if __name__ =="__main__":
   main()
