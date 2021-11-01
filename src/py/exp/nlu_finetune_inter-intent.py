import json
import os

from data import snips
from nlu import load_finetuned
from ufd import Pipeline

report_folder = './reports/global/snips_inter-intent_nlu/'

_, inter_intent_data = snips.get_train_test_data(use_dev=True)

print('Train NLU model')

p = Pipeline(inter_intent_data, dataset_name='inter_intent')
p.train_nlu_model(save_model_path=os.path.join(report_folder, 'inter_intent', 'nlu_model'))

load_finetuned(os.path.join(report_folder, 'inter_intent', 'nlu_model'))
test_quality = json.dumps(p.get_nlu_test_quality(), indent=2)
with open(os.path.join(report_folder, 'inter_intent', 'stats.txt'), 'w') as f:
    f.write(test_quality)
