import json
import statistics
from typing import Callable

import nlu
from data import snips
from ufd import Pipeline


def evaluate_nlu_model_for_support_detection(pipeline: Pipeline, nlu_trained_model_path: str, nst_callback: Callable,
                                             output_file=None):
    nlu.load_finetuned(nlu_trained_model_path)

    feature2label = dict.fromkeys(
        [u.feature_name for u in pipeline.test_utterances if u.feature_name.endswith('_TRAIN')])
    feature2label.update(
        dict.fromkeys([u.feature_name for u in pipeline.test_utterances if u.feature_name.endswith('_DEV')]))
    feature2label.update(
        dict.fromkeys([u.feature_name for u in pipeline.test_utterances if u.feature_name.endswith('_TEST')]))

    details = {}
    for f in feature2label:
        test_ids = [i for i, u in enumerate(pipeline.test_utterances) if u.feature_name == f]

        nlu_stats = pipeline.get_nlu_test_quality(test_ids)
        conf = nlu_stats['conf']
        supported = 1 if nst_callback(conf) else 0
        feature2label[f] = 1 - supported if f.endswith('_TEST') else supported

        utterances = []
        individual_stats = nlu_stats['individual']
        for i, id in enumerate(test_ids):
            utterances.append('{}    ic_conf: {:.3f}    ner_conf: {:.3f}'
                .format(
                ' '.join(['{} ({}{})'.format(t[0], '' if t[1] == 'O' else t[1][:2] + ',', round(t[2], 3))
                          for t in individual_stats['tokens'][i]]),
                individual_stats['conf']['ic'][i],
                individual_stats['conf']['ner_slot'][i]
            ))

        details[f] = {
            'conf': conf,
            'supported': supported,
            'utterances': utterances
        }

    def _quality(labels: list[int]):
        return {
            'total': len(labels),
            'acc': statistics.mean(labels)
        }

    stats = {
        'accuracy': {
            'all': _quality([l for _, l in feature2label.items()]),
            'predict_supported': _quality([l for f, l in feature2label.items() if details[f]['supported'] == 1]),
            'predict_novel': _quality([l for f, l in feature2label.items() if details[f]['supported'] == 0]),
            'TRAIN': _quality([l for f, l in feature2label.items() if f.endswith('_TRAIN')]),
            'DEV': _quality([l for f, l in feature2label.items() if f.endswith('_DEV')]),
            'TRAIN+DEV': _quality([l for f, l in feature2label.items() if not f.endswith('_TEST')]),
            'TEST': _quality([l for f, l in feature2label.items() if f.endswith('_TEST')])
        },
        'details': details
    }

    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(json.dumps(stats, indent=2))

    return stats


_, inter_intent_data = snips.get_train_test_data(use_dev=True)
p = Pipeline(inter_intent_data, dataset_name='inter_intent')

print(json.dumps(
    evaluate_nlu_model_for_support_detection(p, './models/snips_nlu/inter_intent/nlu_model',
                                             nst_callback=lambda conf: conf['ic'] >= 0.9 and conf['ner_slot'] >= 0.9,
                                             output_file='models/snips_nlu/inter_intent/nst.stats.txt'
                                             ),
    indent=2))
