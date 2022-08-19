import re
import pandas as pd
import time
import numpy as np
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.utils.rnn import pad_sequence
from ABSA_SentimentMultiEmiten.model.bert import bert_ABSA

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
    return model

DEVICE = torch.device("cpu")
pretrain_model_name = "indolem/indobert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)
model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
model_path = './models/bert_ABSA_11.pkl'
model_ABSA = load_model(model_ABSA, model_path)

def clean_sentences(sentences, aspect):
  new_sentences =[]
  for sentence in sentences:
    if aspect in sentence:
      new_sentences.append(sentence)
  return new_sentences

## ini kayanya bisa pake regex cuman masi bingung jd pake func dulu :(
def remove_unused_dots(arr_of_char):
  for i in range(len(arr_of_char)):
    if (arr_of_char[i] == "."):
      if(arr_of_char[i - 1].isdigit()):
        arr_of_char[i] == ''
  return "".join(arr_of_char)

def preprocessing_text(sentences, aspect):
  # hapus url
  sentences = re.sub('\S+.com|\S+.co.id|\S+.co|\S+.id', '.', sentences)

  ### hapus titik yang bukan pemisah kalimat 
  ### (yang pemisah angka, Tbk., PT.)
  sentences = sentences.replace("PT.", "PT").replace("Tbk.", "Tbk")
  sentences = remove_unused_dots(sentences)

  ### Ambil kalimat yang mengandung emiten
  sentences = sentences.split('.')
  arr_sentence = clean_sentences(sentences, aspect)
  
  i = 0
  while (i < (len(arr_sentence))):
    arr_sentence[i] = re.sub(r'[^\w\d\s\$\.]+', '', arr_sentence[i]) # Hapus tanda baca kecuali '$'
    arr_sentence[i] = arr_sentence[i].lower()
    ### Menghapus angka
    arr_sentence[i] = ''.join([x for x in arr_sentence[i] if not x.isdigit()])
    i = i+1

  return arr_sentence

def predict(sentence, aspect, tokenizer):
    t1 = tokenizer.tokenize(sentence)
    t2 = tokenizer.tokenize(aspect)

    word_pieces = ['[cls]']
    word_pieces += t1
    word_pieces += ['[sep]']
    word_pieces += t2

    segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)
    segment_tensor = torch.tensor(segment_tensor).to(DEVICE)

    with torch.no_grad():
        outputs = model_ABSA(input_tensor, None, None, segments_tensors=segment_tensor)
        _, predictions = torch.max(outputs, dim=1)
    
    return word_pieces, predictions, outputs