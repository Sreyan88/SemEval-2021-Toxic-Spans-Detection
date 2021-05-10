# Cisco-at-SemEval-2021-Task-5-Toxic-Spans-Detection
This repository contains code for our paper titled : Cisco at SemEval-2021 Task 5: What’s Toxic?: Leveraging Transformers for Multiple Toxic Span Extraction from Online Comments

The paper was accepted at SemEval 2021 and is based on the shared task SemEval 2021 Task 5 : Toxic Spans Detection


We provide the [link](https://drive.google.com/drive/folders/1NtNUmLs9rgdpAkSgxzhPdSQa3yOqjvcS?usp=sharing) to one of our *Sequence Tagging Model (BERT Model trained on BIO Tagging)*.

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

