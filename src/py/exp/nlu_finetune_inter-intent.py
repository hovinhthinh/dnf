import json
import os

from transformers import set_seed

from data import snips
from nlu import load_finetuned
from ufd import Pipeline

report_folder = './models/snips_nlu/'

_, inter_intent_data = snips.get_train_test_data(use_dev=True)

print('Train NLU model')
set_seed(12993)

p = Pipeline(inter_intent_data, dataset_name='inter_intent')
# p.train_nlu_model(save_model_path=os.path.join(report_folder, 'inter_intent', 'nlu_model'), n_train_epochs=10)

load_finetuned(os.path.join(report_folder, 'inter_intent', 'nlu_model'))
test_quality = json.dumps(p.get_nlu_test_quality(), indent=2)
with open(os.path.join(report_folder, 'inter_intent', 'stats.txt'), 'w') as f:
    print(test_quality)
    f.write(test_quality)
