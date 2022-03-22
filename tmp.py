from transformers import RobertaModel, RobertaTokenizer

m = RobertaModel.from_pretrained('roberta-base')
t = RobertaTokenizer.from_pretrained('roberta-base')

i = 'hello my name is'

i = t(i, return_tensors='pt')
output = m(**i)