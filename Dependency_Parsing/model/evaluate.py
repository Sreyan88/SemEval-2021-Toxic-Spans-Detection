#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import tensorflow as tf
import tensorflow.compat.v1 as tf
import json
import sys
tf.compat.v1.disable_v2_behavior()

import util,biaffine_ner_model

if __name__ == "__main__":
  config = util.initialize_from_env()

  config['eval_path'] = config['test_path']

  model = biaffine_ner_model.BiaffineNERModel(config)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True
  with tf.Session(config=session_config) as session:
    model.restore(session)
    _,_,pred_ners,gold_ners = model.evaluate(session,True)
    #_1000 added for data given by Sonal
    with open(sys.argv[1]+'_test_preds_1000.json','w') as f1 , open(sys.argv[1]+'_test_reals_1000.json','w') as f2:
      json.dump(list(pred_ners),f1)
      json.dump(list(gold_ners),f2)

