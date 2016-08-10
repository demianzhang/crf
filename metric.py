from reader import labels_to_id as labels
bset = set()
iset = set()
oset = set()
bset.add(labels['B-PLACE'])
bset.add(labels['B-PERSON'])
bset.add(labels['B-ORGANIZATION'])
bset.add(labels['B-OTHERPROPNOUN'])
iset.add(labels['I-PLACE'])
iset.add(labels['I-PERSON'])
iset.add(labels['I-ORGANIZATION'])
iset.add(labels['I-OTHERPROPNOUN'])
oset.add(labels['O'])

b_i = dict([(labels['B-PLACE'], labels['I-PLACE']), 
          (labels['B-PERSON'], labels['I-PERSON']), 
          (labels['B-ORGANIZATION'], labels['I-ORGANIZATION']), 
          (labels['B-OTHERPROPNOUN'], labels['I-OTHERPROPNOUN'])])

def precision_recall(pred, pred_length, truth, truth_length,
    bset=bset, iset=iset, oset=oset):
  truth_total = 0
  recall = 0.0
  pred_total = 0
  precision = 0.0

  for k, (_pred, _truth) in enumerate(zip(pred, truth)):
    if pred_length != None and truth_length != None:
      assert(pred_length[k] == truth_length[k])
      _length = pred_length[k]
    else:
      assert(len(_pred) == len(_truth))
      _length = len(_pred)

    # recall
    found = 0
    i = 0
    while i < _length:
      if _truth[i] in bset:
        found += 1
      i += 1 # iset, oset

    truth_total += found

    # precision
    # for each BI sequence in _pred, test if it's a match with _truth
    found = 0
    match = 0
    i = 0
    while i < _length:
      if _pred[i] in bset:
        tag_i_for_b = b_i[_pred[i]] # expect label i for this maching
        found += 1
        if _truth[i] == _pred[i]: # start matching
          i += 1
          while i < _length and _truth[i] in iset and _truth[i] == _pred[i]:
            i += 1
          if i == _length or (_truth[i] not in iset and _pred[i] != tag_i_for_b):
            match += 1
          continue
      i += 1 # iset, oset

    pred_total += found
    precision += match

  recall = precision
  pvalue = precision / pred_total if pred_total > 0 else 0.0
  rvalue = recall / truth_total if truth_total > 0 else 0.0
  f1_score = 0 if pvalue + rvalue == 0 else 2 * (pvalue * rvalue) / (pvalue + rvalue)
  print("precision: {} of total {}".format(pvalue, pred_total))
  print("recall: {} of total {}".format(rvalue, truth_total))
  print("f1_score: {}".format(f1_score))
  print("")
  return precision, pred_total, recall, truth_total, f1_score


if __name__ == '__main__':
  p=[[1, 0, 0]]
  t=[[1, 0, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 1 and pre_t == 1 and rec == 1 and tt == 1)

  p=[[1, 0, 0]]
  t=[[0, 0, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 0 and pre_t == 1 and rec == 0 and tt == 0)

  p=[[1, 2, 0, 0]]
  t=[[1, 2, 0, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 1 and pre_t == 1 and rec == 1 and tt == 1)

  p=[[1, 0, 0, 0]]
  t=[[1, 2, 0, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 0 and pre_t == 1 and rec == 0 and tt == 1)

  p=[[0, 2, 0, 0]]
  t=[[1, 2, 0, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 0 and pre_t == 0 and rec == 0 and tt == 1)

  p=[[1, 2, 0, 0]]
  t=[[1, 0, 0, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 0 and pre_t == 1 and rec == 0 and tt == 1)

  p=[[1, 2, 4, 0]]
  t=[[1, 2, 0, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 1 and pre_t == 1 and rec == 1 and tt == 1)

  p=[[1, 2, 0, 0]]
  t=[[1, 2, 2, 0]]
  pre, pre_t, rec, tt = precision_recall(p, None, t, None)
  assert(pre == 0 and pre_t == 1 and rec == 0 and tt == 1)