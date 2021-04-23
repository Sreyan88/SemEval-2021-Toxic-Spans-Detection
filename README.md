# Cisco-at-SemEval-2021-Task-5-Toxic-Spans-Detection
This repository contains code for our paper titled : Cisco at SemEval-2021 Task 5: What’s Toxic?: Leveraging Transformers for Multiple Toxic Span Extraction from Online Comments

The paper was accepted at SemEval 2021 and is based on the shared task SemEval 2021 Task 5 : Toxic Spans Detection


We provide the link to our Sequence Tagging (BERT Model trained on BIO Tagging) - https://drive.google.com/drive/folders/1NtNUmLs9rgdpAkSgxzhPdSQa3yOqjvcS?usp=sharing

For tagging your own text download the model from drive and run:

```python
from flair.models import SequenceTagger
from flair.data import Sentence

tagger = SequenceTagger.load('/path/to/your/model')

sentence = Sentence('He is so stupid !')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tagged_string())
```

