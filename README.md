# lstm-crf
BiLSTM + linear chain CRF for sequence labeling in Chinese Social Media

charpos embedding:  
> Hangfeng He and Xu Sun  
> F-Score Driven Max Margin Neural Network for Named Entity Recognition in Chinese Social Media(CoRR),2016   

> Nanyun Peng and Mark Dredze  
> Conference on Empirical Methods in Natural Language Processing (EMNLP), 2015  

## Dependencies:
This is an theano implementation; it requires installation of python module:  
> Theano  

It can be simply installed by pip module Name.  

## How to run：  
```shell
python train_emb.py
```  

## Model  
```bash  
dropout_layer -> lstm_layer -> tanh -> fc_layer -> crf  
MomentumSGD | RMSprop
```
## Result on test
```bash
precision: 0.675  
recall: 0.486  
f1: 0.563
```
