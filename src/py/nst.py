import json
import os
import statistics
from typing import Callable

import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

import nlu
from data import snips
from sbert import PseudoClassificationModel, get_pseudo_classifier_confidence, \
    _split_text_and_slots_into_tokens_and_tags
from ufd import Pipeline


def _populate_pr_auc_results(metrics, novelty, conf, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    pr_auc_results = {}

    baseline_prec = sum(novelty) / len(novelty)
    for m in metrics:
        prec, rec, threshold = precision_recall_curve(novelty, [-c[m] for c in conf])
        f1 = [0 if prec[i] + rec[i] == 0 else 2 * prec[i] * rec[i] / (prec[i] + rec[i]) for i in range(len(prec))]

        best_id = numpy.argmax(numpy.asarray(f1))
        pr_auc_results[m] = {
            'pr_auc_score': round(auc(rec, prec), 3),
            'optimal_f1': round(f1[best_id], 3),
            'threshold': round(-threshold[best_id], 3),
            'prec': round(prec[best_id], 3),
            'rec': round(rec[best_id], 3),
        }

        # Plot PR curve
        plt.plot([0, 1], [baseline_prec, baseline_prec], linestyle='--', label='random')  # Baseline
        plt.plot(rec, prec, marker='.', linestyle='-', label='conf: {}'.format(m))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend()
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'pr_curve.{}.pdf'.format(m)), bbox_inches='tight')
        plt.close()

    with open(os.path.join(output_folder, 'stats.txt'), 'w') as f:
        f.write(json.dumps(pr_auc_results, indent=2))

    return pr_auc_results


def _update_conf(conf):
    conf['ic x ner_tag_min'] = conf['ic'] * conf['ner_tag_min']
    conf['ner_tag_min x pc'] = conf['pc'] * conf['ner_tag_min']
    conf['ic x ner_tag_min x pc'] = conf['ic'] * conf['ner_tag_min'] * conf['pc']
    # conf.pop('ic')


def compute_pr_auc_for_nlu_model_for_support_detection(pipeline: Pipeline, nlu_trained_model_path: str,
                                                       pseudo_classification_trained_model_path: str,
                                                       output_folder: str):
    nlu.load_finetuned(nlu_trained_model_path)

    # Cluster level
    feature2novel = {u.feature_name: 0 for u in pipeline.test_utterances if u.feature_name.endswith('_TRAIN')}
    feature2novel.update({u.feature_name: 0 for u in pipeline.test_utterances if u.feature_name.endswith('_DEV')})
    feature2novel.update({u.feature_name: 1 for u in pipeline.test_utterances if u.feature_name.endswith('_TEST')})

    pc = PseudoClassificationModel.load_model(pseudo_classification_trained_model_path)
    feature2conf = {}

    for f in feature2novel:
        test_ids = [i for i, u in enumerate(pipeline.test_utterances) if u.feature_name == f]

        nlu_stats = pipeline.get_nlu_test_quality(test_ids)
        conf = nlu_stats['conf']
        pc_conf = get_pseudo_classifier_confidence([pipeline.test_utterances[i].text for i in test_ids], pc)
        conf['pc'] = statistics.mean(pc_conf).item()
        _update_conf(conf)
        feature2conf[f] = conf

    # Utterance level
    utterance_novelty = [1 if u.feature_name.endswith('_TEST') else 0 for u in pipeline.test_utterances]
    individual_conf = pipeline.get_nlu_test_quality(list(range(len(pipeline.test_utterances))))['individual']['conf']
    pc_conf = get_pseudo_classifier_confidence([u.text for u in pipeline.test_utterances], pc)
    utterance_conf = []
    for i in range(len(pipeline.test_utterances)):
        conf = {k: float(v[i]) for k, v in individual_conf.items()}
        conf['pc'] = float(pc_conf[i])
        _update_conf(conf)
        utterance_conf.append(conf)

    _populate_pr_auc_results(conf.keys(), [feature2novel[f] for f in feature2novel],
                             [feature2conf[f] for f in feature2novel],
                             os.path.join(output_folder, 'cluster_level'))
    _populate_pr_auc_results(conf.keys(), utterance_novelty, utterance_conf,
                             os.path.join(output_folder, 'utterance_level'))


def compute_dev_pr_auc_for_nlu_model_for_support_detection(pipeline: Pipeline, nlu_trained_model_path: str,
                                                           pseudo_classification_trained_model_path: str,
                                                           output_folder: str):
    nlu.load_finetuned(nlu_trained_model_path)

    # Cluster level
    feature2novel = {u.feature_name: 0 for u in pipeline.utterances if
                     u.feature_name.endswith('_TRAIN') and u.part_type == 'DEV'}
    feature2novel.update(
        {u.feature_name: 1 for u in pipeline.utterances if u.feature_name.endswith('_DEV') and u.part_type == 'DEV'})

    pc = PseudoClassificationModel.load_model(pseudo_classification_trained_model_path)
    feature2conf = {}

    def _get_nlu_dev_quality(utterances):
        texts, tags = _split_text_and_slots_into_tokens_and_tags([u.text for u in utterances],
                                                                 [u.slots for u in utterances])
        nlu_outputs = nlu.get_intents_and_slots(texts)
        return pipeline._get_nlu_quality(nlu_outputs, [u.intent_name for u in utterances], tags,
                                         keep_individual_stats=True)

    for f in feature2novel:
        utterances = [u for u in pipeline.utterances if u.feature_name == f and u.part_type == 'DEV']
        nlu_stats = _get_nlu_dev_quality(utterances)
        conf = nlu_stats['conf']
        pc_conf = get_pseudo_classifier_confidence([u.text for u in utterances], pc)
        conf['pc'] = statistics.mean(pc_conf).item()
        _update_conf(conf)
        feature2conf[f] = conf

    # Utterance level
    utterance_novelty = [1 if u.feature_name.endswith('_DEV') else 0 for u in pipeline.utterances if
                         u.part_type == 'DEV']
    utterances = [u for u in pipeline.utterances if u.part_type == 'DEV']
    individual_conf = _get_nlu_dev_quality(utterances)['individual']['conf']
    pc_conf = get_pseudo_classifier_confidence([u.text for u in utterances], pc)
    utterance_conf = []
    for i in range(len(utterances)):
        conf = {k: float(v[i]) for k, v in individual_conf.items()}
        conf['pc'] = float(pc_conf[i])
        _update_conf(conf)
        utterance_conf.append(conf)

    _populate_pr_auc_results(conf.keys(), [feature2novel[f] for f in feature2novel],
                             [feature2conf[f] for f in feature2novel],
                             os.path.join(output_folder, 'cluster_level'))
    _populate_pr_auc_results(conf.keys(), utterance_novelty, utterance_conf,
                             os.path.join(output_folder, 'utterance_level'))


def evaluate_nlu_model_for_support_detection(pipeline: Pipeline, nlu_trained_model_path: str,
                                             pseudo_classification_trained_model_path: str,
                                             nst_callback: Callable,
                                             output_file=None):
    nlu.load_finetuned(nlu_trained_model_path)

    feature2label = dict.fromkeys(
        [u.feature_name for u in pipeline.test_utterances if u.feature_name.endswith('_TRAIN')])
    feature2label.update(
        dict.fromkeys([u.feature_name for u in pipeline.test_utterances if u.feature_name.endswith('_DEV')]))
    feature2label.update(
        dict.fromkeys([u.feature_name for u in pipeline.test_utterances if u.feature_name.endswith('_TEST')]))

    pc = PseudoClassificationModel.load_model(pseudo_classification_trained_model_path) \
        if pseudo_classification_trained_model_path is not None else None
    details = {}
    for f in feature2label:
        test_ids = [i for i, u in enumerate(pipeline.test_utterances) if u.feature_name == f]

        nlu_stats = pipeline.get_nlu_test_quality(test_ids)
        conf = nlu_stats['conf']

        if pc is not None:
            pc_conf = get_pseudo_classifier_confidence([pipeline.test_utterances[i].text for i in test_ids], pc)
            conf['pc'] = statistics.mean(pc_conf).item()
            _update_conf(conf)
        supported = 1 if nst_callback(conf) else 0
        feature2label[f] = 1 - supported if f.endswith('_TEST') else supported

        utterances = []
        individual_stats = nlu_stats['individual']
        for i, id in enumerate(test_ids):
            utterances.append('{}    ic_conf: {:.3f}    ner_conf: {:.3f}    pc_conf: {:.3f}'
                .format(
                ' '.join(['{} ({}{})'.format(t[0], '' if t[1] == 'O' else t[1][:2] + ',', round(t[2], 3))
                          for t in individual_stats['tokens'][i]]),
                individual_stats['conf']['ic'][i],
                individual_stats['conf']['ner_slot'][i],
                pc_conf[i].item() if pc is not None else None
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


if __name__ == '__main__':
    _, inter_intent_data = snips.get_train_test_data(use_dev=True)
    p = Pipeline(inter_intent_data, dataset_name='inter_intent')

    # compute_dev_pr_auc_for_nlu_model_for_support_detection(p, './models/snips_nlu_exclude_unseen/inter_intent/nlu_model',
    #                                                        './reports/global/snips_SMC+US_PC_exclude_unseen/inter_intent/pc_trained_model',
    #                                                        './reports/nst/nlu_validation/snips_inter_intent_exclude_unseen/')
    # compute_pr_auc_for_nlu_model_for_support_detection(p, './models/snips_nlu/inter_intent/nlu_model',
    #                                                    './reports/global/snips_SMC+US_PC/inter_intent/pc_trained_model',
    #                                                    './reports/nst/nlu_validation/snips_inter_intent/')
    print(evaluate_nlu_model_for_support_detection(p, './models/snips_nlu/inter_intent/nlu_model',
                                                   './reports/global/snips_SMC+US_PC/inter_intent/pc_trained_model',
                                                   nst_callback=lambda conf: conf['ic x ner_tag_min x pc'] > 0.5,
                                                   output_file='./reports/nst/nlu_validation/snips_inter_intent/cluster_stats_ic x ner_tag_min x pc.txt'
                                                   ))
