import os
import codecs as cs
#os.environ["THEANO_FLAGS"] = 'floatX=float32,device=gpu1,lib.cnmem=0'
os.environ["THEANO_FLAGS"] = 'floatX=float32,lib.cnmem=0,optimizer=fast_compile'

from nn import Embedding, Dropout, FullConnect, LSTM, BiLSTM, CRF, RMSprop, MomentumSGD
import reader
import metric
import sys
import numpy
import theano
import pickle
from theano import tensor

class CRF_MODEL():
  def __init__(self, fn, dic, hidden_size, output_size):
    X = tensor.ivector()
    Y = tensor.ivector()
    keep_prob = tensor.fscalar()
    learning_rate = tensor.fscalar()

    emb_layer = Embedding(fn, dic)
    lstm_layer = BiLSTM(hidden_size, hidden_size)
    dropout_layer = Dropout(keep_prob)
    fc_layer = FullConnect(2*hidden_size, output_size)
    crf = CRF(output_size)
    # graph defination
    X_emb = emb_layer(X)
    scores = fc_layer(tensor.tanh(lstm_layer(dropout_layer(X_emb))))
    
    loss, predict = crf(scores, Y, isTraining=True)
    # loss, predict and accuracy
    accuracy = tensor.sum(tensor.eq(predict, Y)) * 1.0 / Y.shape[0]

    params = emb_layer.params + lstm_layer.params + fc_layer.params + crf.params
    updates = MomentumSGD(loss, params, lr=learning_rate)

    print("Compiling train function: ")
    train = theano.function(inputs=[X, Y, keep_prob, learning_rate], outputs=[predict, accuracy, loss], 
      updates=updates, allow_input_downcast=True)

    print("Compiling evaluate function: ")
    evaluate = theano.function(inputs=[X_emb, Y, keep_prob], outputs=[predict, accuracy, loss], 
      allow_input_downcast=True)

    self.embedding_tensor = emb_layer.params[0]
    self.train = train
    self.evaluate = evaluate
    self.params = params

  def save(self, file_path):
    with open(file_path, 'wb') as fout:
      params_values = [p.get_value() for p in self.params]
      pickle.dump(params_values, fout)

  def load(self, file_path):
    with open(file_path, 'rb') as fin:
      params_values = pickle.load(fin)
      for p, v in zip(self.params, params_values):
        p.set_value(v)


# helper function for evaluation
def average_embedding(embeddings, word_id_list):
    emb = [0] * len(word_id_list)
    for i,v in enumerate(word_id_list):
      if v >= 0:
        emb[i] = embeddings[v]

    for i,v in enumerate(word_id_list):
      if v < 0:
        emb[i] = numpy.zeros(embeddings.shape[1])
        count = 0
        for k in range(max(0, i-4), i):
          if word_id_list[k] >= 0:
            count += 1
            emb[i] += emb[k]
        for k in range(i+1, min(i+5, len(word_id_list))):
          if word_id_list[k] >= 0:
            count += 1
            emb[i] += emb[k]
        if count > 0:
          emb[i] = emb[i] / count
    return numpy.asarray(emb, dtype=numpy.float32)

def write_prediction(filename, output_dir, lex_test, pred_test):
    with cs.open(os.path.join(os.path.basename(output_dir), filename), 'w', encoding='utf-8') as outf:
        for sent_w, sent_l in zip(lex_test, pred_test):
            assert len(sent_w) == len(sent_l)
            for w, l in zip(sent_w, sent_l):
                outf.write(w+'\t'+l+'\n')
            outf.write('\n')

def convert_id_to_word(corpus, idx2label):
    return [[idx2label[word] for word in sentence] for sentence in corpus]

def main():
  train_x, train_y, test_x, test_y, word_to_id, labels = reader.load()
  # id_to_word = dict((k, v) for v, k in word_to_id.iteritems())
  # lex_test = convert_id_to_word(test_x,id_to_word)
  m = CRF_MODEL('weibo_charpos_vectors', word_to_id, 100, len(labels))

  # m.load("checkpoints_emb/crf_emb_14.0.8326.pkl")
  lr = 0.001
  best_cv_f1_score = 0.832
  for epoch in range(0, 25):
    print("epoch: ", epoch)
    if epoch > 0 and epoch % 10 == 0:
      lr = lr / 10

    print("learning_rate: ", lr)

    accu = 0
    loss = 0
    pred_list = []
    truth_list = []

    perm = numpy.random.permutation(numpy.arange(len(train_x)))
    for i in perm:
      _pred, _accu, _loss = m.train(train_x[i], train_y[i], 0.5, lr)
      accu += _accu
      loss += _loss
      pred_list.append(_pred)
      truth_list.append(train_y[i])

    # print("train_hits: {}  train_loss: {} ".format(accu/2000, loss/2000))
    print("train_hits: {}  train_loss: {} ".format(accu/len(train_x), loss/len(train_x)))
    metric.precision_recall(pred_list, None, truth_list, None)
    print("")


    embeddings = m.embedding_tensor.get_value()
    accu = 0
    loss = 0
    pred_list = []
    for _x, _y in zip(test_x, test_y):
      _x_emb = average_embedding(embeddings, _x)
      _pred, _accu, _loss = m.evaluate(_x_emb, _y, 1.0)
      accu += _accu
      loss += _loss
      #print _pred
      #print '\n'
      pred_list.append(_pred)
    print("cv_hits: {}  cv_loss: {} ".format(accu/len(test_x), loss/len(test_x)))
    _,_,_,_, _f1 = metric.precision_recall(pred_list, None, test_y, None)
    print("")
    
    # id_to_word = dict((k, v) for v, k in word_to_id.iteritems())
    # lex_test = convert_id_to_word(test_x,id_to_word)
    # lex_pred_list = convert_id_to_word(pred_list,id_to_labels)
    # write_prediction('weiboNER.conll.test'+'.prediction','/data/',lex_test, lex_pred_list)

    if _f1 > best_cv_f1_score:
      best_cv_f1_score = _f1
      m.save("./checkpoints_emb/{}_{}_{:.4f}.pkl".format("crf_emb", epoch, _f1))

if __name__ == '__main__':
  main()
