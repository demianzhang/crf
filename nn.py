import numpy
import os
import codecs 
#os.environ["THEANO_FLAGS"] = 'floatX=float32,device=gpu0,lib.cnmem=0'
#os.environ["THEANO_FLAGS"] = 'floatX=float32,device=gpu1,lib.cnmem=0'
import theano
from theano import shared
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

# global settings for code convinence
SEED = 1024
trng = RandomStreams(SEED)
floatX = theano.config.floatX


def uniform_init(shape):
  return numpy.random.uniform(low=-0.1, high=0.1, size=shape).astype(floatX)

def read_matrix_from_file(fn, dic, ecd='utf-8'):
    multiplier = numpy.sqrt(1.0/3)
    not_in_dict = 0
    with codecs.open(fn, encoding=ecd, errors='ignore') as inf:
        row, column = inf.readline().rstrip().split()
        dim = int(column)
        #print row, column
        idx_map = dict()
        line_count = 0
        M = uniform_init((len(dic), dim))
        for line in inf:
            elems = line.rstrip().split(' ')
            if elems[0] in dic:
                idx = dic[elems[0]]
                vec_elem = elems[1:]
                r = numpy.array([float(_e) for _e in vec_elem])
                M[idx] = (r/numpy.linalg.norm(r)) * multiplier 
                idx_map[idx] = line_count
            else:
                not_in_dict += 1
            line_count += 1
        print 'load embedding! %s words, %d not in the dictionary. Dictionary size: %d' %(row, not_in_dict, len(dic))
        #print M.shape, len(idx_map)
        return M

class Abc():
  def __init__(self):
    self._params = []

  @property
  def params(self):
    if isinstance(self._params, list):
      return self._params
    return [self._params]


class Embedding(Abc):
  def __init__(self, fn, dic):
    self.W = shared(read_matrix_from_file(fn, dic))
    # self.W = shared(uniform_init((input_size, embedding_size)))
    self._params = self.W

  def __call__(self, input):
    # return a sub_tensor
    return self.W[input]


class Dropout():
  def __init__(self, keep_prob):
    self.keep_prob = keep_prob

  def __call__(self, prev):
    proj = tensor.switch( self.keep_prob < 1.0,
        (prev * trng.binomial(prev.shape, p=self.keep_prob, n=1, 
                dtype=prev.dtype) * (1/self.keep_prob)),
        prev)
    return proj


class FullConnect(Abc):
  def __init__(self, input_size, output_size):
    self.W = shared(uniform_init((input_size, output_size)))
    self.b = shared(uniform_init((output_size)))
    self._params = [self.W, self.b]

  def __call__(self, input):
    return tensor.dot(input, self.W) + self.b


class LSTM(Abc):
  def __init__(self, input_size, hidden_size):
    self.hidden_size = hidden_size
    self.W = shared(uniform_init((input_size+hidden_size+1, 4*hidden_size)))
    self._params = self.W

  def __call__(self, input):
    nh = self.hidden_size
    # _in: input of t
    # _m : output of t - 1
    # _c : memory of t - 1 
    def _step(_in, _m, _c, nh):
      _x = tensor.concatenate([numpy.asarray([1.], dtype=numpy.float32), _in, _m])
      ifog = tensor.dot(_x, self.W)

      # i = tensor.nnet.sigmoid(ifog[ : nh])
      # f = tensor.nnet.sigmoid(ifog[nh : 2*nh])
      # o = tensor.nnet.sigmoid(ifog[2*nh : 3*nh])
      # g = tensor.tanh(ifog[3*nh : ])
      
      i = tensor.nnet.sigmoid(ifog[ : 100])
      f = tensor.nnet.sigmoid(ifog[100 : 200])
      o = tensor.nnet.sigmoid(ifog[200 : 300])
      g = tensor.tanh(ifog[300 : 400])

      _c = f * _c + i * g
      _m = o * _c
      return _m, _c
    self._step = _step

    results, update = theano.scan(
        _step, 
        sequences=[input],
        outputs_info=[tensor.alloc(0.0, nh), tensor.alloc(0.0, nh)],
        non_sequences=[self.hidden_size]
      )
    return results[0] #(_m_list, _c_list)[0]

  # it's useful in sampling
  def one_step(self, x, m, c):
    return self._step(x, m, c, self.hidden_size)

class BiLSTM(Abc):
  def __init__(self, input_size, hidden_size):
    self.forward = LSTM(input_size, hidden_size)
    self.backward = LSTM(input_size, hidden_size)
    self._params = self.forward.params + self.backward.params

  def __call__(self, input):
    f_result = self.forward(input)
    b_result = self.backward(input[::-1])
    return tensor.concatenate([f_result, b_result[::-1]], axis=1)


class CRF(Abc):
  def __init__(self, tag_num):
    # add a start-tag and an end-tag in the transition matrix
    self.transitions = shared(uniform_init((tag_num+2, tag_num+2)))
    self.tag_num = tag_num
    self._params = self.transitions

  def __call__(self, input, labels=None, isTraining=False):
    small = -1000
    b_padding = numpy.array([[small] * self.tag_num + [0, small]]).astype(floatX)
    e_padding = numpy.array([[small] * self.tag_num + [small, 0]]).astype(floatX)
    input_padded = tensor.concatenate(
        [input, small * tensor.ones((input.shape[0], 2))],
        axis = 1
      )
    input_padded = tensor.concatenate(
        [b_padding, input_padded, e_padding],
        axis = 0
      )

    def log_sum_exp(x):
      # https://en.wikipedia.org/wiki/LogSumExp
      xmax = x.max(axis=0, keepdims=True)
      xmax_ = x.max(axis=0)
      return xmax_ + tensor.log(tensor.exp(x - xmax).sum(axis=0))

    def _step(curr, prev, tran):
      prev = prev.dimshuffle(0, 'x')
      curr = curr.dimshuffle('x', 0)
      x = prev + curr + tran
      return log_sum_exp(x)

    allpath, _ = theano.scan(
        fn = _step,
        outputs_info = input_padded[0],
        sequences=[input_padded[1:]],
        non_sequences=self.transitions
      )
    if isTraining:
      # [arxiv:1603.01360]
      # loss = -(realpath - allpath)
      realpath = (input[tensor.arange(labels.shape[0]), labels]).sum()
      
      b_id = theano.shared(value=numpy.array([self.tag_num], dtype=numpy.int32))
      e_id = theano.shared(value=numpy.array([self.tag_num + 1], dtype=numpy.int32))
      padded_tags_ids = tensor.concatenate([b_id, labels, e_id], axis=0)
      realpath += (self.transitions[
          padded_tags_ids[tensor.arange(labels.shape[0] + 1)],
          padded_tags_ids[tensor.arange(labels.shape[0] + 1)+1]
        ]).sum()
      
      loss = -(realpath - log_sum_exp(allpath[-1]))
  
    def _step_best(curr, prev, tran):
      prev = prev.dimshuffle(0, 'x')
      curr = curr.dimshuffle('x', 0)
      x = prev + curr + tran
      return x.max(axis=0), x.argmax(axis=0)

    bestpath_weights, _ = theano.scan(
        fn = _step_best,
        outputs_info = (input_padded[0], None),
        sequences = [input_padded[1:]],
        non_sequences = self.transitions
      )
    sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=tensor.cast(tensor.argmax(bestpath_weights[0][-1]), 'int32'),
            sequences=tensor.cast(bestpath_weights[1][::-1], 'int32')
        )
    # predict = tensor.concatenate([sequence[::-1], [tensor.argmax(bestpath_weights[0][-1])]])
    predict = sequence[-2::-1] # without start and end tag!
    if isTraining :
      return loss, predict
    else:
      return predict


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
  grads = tensor.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
      acc = theano.shared(p.get_value() * 0.)
      acc_new = rho * acc + (1 - rho) * g ** 2
      gradient_scaling = tensor.sqrt(acc_new + epsilon)
      g = g / gradient_scaling
      updates.append((acc, acc_new))
      updates.append((p, p - lr * g))
  return updates

def MomentumSGD(cost, params, lr=0.001, momentum=0.9):
  grads = tensor.grad(cost=cost, wrt=params)
  update_list = []
  for p, g in zip(params, grads):  
    value = p.get_value(borrow=True)
    velocity = theano.shared(numpy.zeros(value.shape, dtype=value.dtype))
    update_list.append((velocity, momentum * velocity + lr * g))
    update_list.append((p, p - velocity))

  return OrderedDict(update_list)