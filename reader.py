import numpy
import os
import codecs as cs
import pickle

labels = ['O', 'B-PER.NOM', 'I-PER.NOM','B-PER.NAM', 'I-PER.NAM', 'B-ORG.NAM',
    'I-ORG.NAM', 'B-GPE.NAM', 'I-GPE.NAM','B-LOC.NAM','I-LOC.NAM','B-LOC.NOM','I-LOC.NOM',
    'B-ORG.NOM','I-ORG.NOM','B-GPE.NOM','I-GPE.NOM']
id_to_labels = dict(enumerate(labels))
labels_to_id = dict((v,i) for i, v in enumerate(labels))

def load(data_file = "ner_data2.pkl"):
  if os.path.exists(data_file):
    return pickle.load(open(data_file, 'rb'))

  word_set = set()
  def load_file(fn, build_dict=True,has_label=True):
    with cs.open(fn, encoding='utf-8') as src:
        stream = src.read().strip().split('\n\n')
        corpus = []
        labels = []
        for line in stream:
            line = line.strip().split('\n')
            sentc = []
            label = []
            for e in line:
                token = e.split()
                sentc.append(token[0])
                if has_label:
                    label.append(labels_to_id[token[-1]])
                else:
                    label.append(None)
                if build_dict:
                    word_set.add(token[0])
            corpus.append(sentc)
            labels.append(label)
    return corpus, labels
  train_x, train_y = load_file('weiboNER_2nd_conll.train', build_dict=True, has_label=True)
  test_x, test_y = load_file('weiboNER_2nd_conll.dev')
  word_to_id = dict((w, i) for i, w in enumerate(word_set))
  print('total words: ', len(word_to_id))
  print('train set size: ', len(train_x))
  print('test set size: ', len(test_x))
  

  def convert(X): 
    Xid = []
    for _x in X:
      Xid.append([word_to_id[v] if v in word_to_id else -1 for v in _x])
    return Xid

  trainX = numpy.asarray(convert(train_x))
  trainY = numpy.asarray(train_y)
  testX = numpy.asarray(convert(test_x))
  testY = numpy.asarray(test_y)
  with open(data_file, 'wb') as fout:
    pickle.dump((trainX, trainY, testX, testY, word_to_id, labels_to_id), fout)

  return trainX, trainY, testX, testY, word_to_id, labels_to_id

