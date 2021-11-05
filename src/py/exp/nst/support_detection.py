import json

import nlu
from data import snips
from ufd import Pipeline

# Load NLU model
nlu.load_finetuned('./models/snips_nlu/inter_intent/nlu_model')

_, inter_intent_data = snips.get_train_test_data(use_dev=True)

p = Pipeline(inter_intent_data, dataset_name='inter_intent')

# Evaluate predicted clusters
predicted_clusters = json.loads(
    open('reports/global/snips_SMC+US_PC_elbow/inter_intent/2_PC_test.predicted_clusters.stats.json', 'r').read())

clusters = predicted_clusters.pop('clusters')


def supported(conf):
    return conf['intent'] >= 0.9 and conf['slot'] >= 0.9


with open('reports/nst/snips_inter_intent_stats.txt', 'w') as f:
    f.write(f'{predicted_clusters}\n')

    for c in clusters:
        nlu_quality = p.get_nlu_test_quality(c.pop('test_utterance_ids'))
        nlu_quality.pop('individual')
        nlu_conf = nlu_quality.pop('conf')
        cluster_quality = c.pop('quality')

        f.write('\n\n')
        f.write(f'stats: {c}\n')
        f.write(f'cluster_quality: {cluster_quality}\n')
        f.write(f'nlu_confidence: {nlu_conf} => {"SUPPORTED" if supported(nlu_conf) else "NOVEL"}\n')
        f.write(f'nlu_quality: {nlu_quality}\n')

        f.flush()
