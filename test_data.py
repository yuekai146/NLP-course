from data import *

e_gen = example_generator("/home/zhaoyuekai/torch_code/data/summary/finished_files/chunked/train_*", False)
e = next(e_gen)
article_text = e.features.feature['article'].bytes_list.value[0]
abstract_text = e.features.feature['abstract'].bytes_list.value[0]
abstract_text = str(abstract_text)
abs_sents = abstract2sents(abstract_text)
print(abs_sents)