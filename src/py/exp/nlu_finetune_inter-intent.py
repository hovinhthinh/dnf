import json
import os

from data import snips
from nlu import load_finetuned
from ufd import Pipeline

report_folder = './reports/global/snips_inter-intent_nlu/'

_, inter_intent_data = snips.get_train_test_data(use_dev=True)

print('Train NLU model')

p = Pipeline(inter_intent_data, dataset_name='inter_intent')
# p.train_nlu_model(save_model_path=os.path.join(report_folder, 'inter_intent', 'nlu_model'), n_train_epochs=5)

load_finetuned('models/snips_inter-intent_nlu/inter_intent/nlu_model')
print(json.dumps(p.get_nlu_test_quality(), indent=2))
