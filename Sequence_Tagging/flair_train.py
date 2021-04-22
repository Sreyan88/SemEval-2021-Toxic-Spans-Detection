import flair
import torch
import argparse

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input', '-i',
                        help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output', '-o',
                        help='Name of the output folder')
parser.add_argument('--gpu', '-g',
                        help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')


args = parser.parse_args()
input_folder=args.input
output_folder=args.output
gpu_type=args.gpu


flair.device = torch.device(gpu_type)
from typing import List
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import *
# from flair.embeddings import TransformerWordEmbeddings

# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

tag_type = 'ner'

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file='train.txt',
                              dev_file='dev.txt',test_file = 'test.txt')

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types: List[TokenEmbeddings] = [

     TransformerWordEmbeddings('xlnet-large-cased',fine_tune = True),
     CharacterEmbeddings()
 ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# trainer.train(output_folder, learning_rate=0.01,
#               mini_batch_size=64,
#               max_epochs=150)
#from pathlib import Path

# Load from checkpoint
#checkpoint = '/media/data_dump/sreyan/semeval/output_final_xlnet_character/checkpoint.pt'
#trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)

trainer.train(output_folder, learning_rate=0.01,
              mini_batch_size=8,
              max_epochs=150,embeddings_storage_mode='gpu')
