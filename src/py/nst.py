import json
import os
import statistics
from typing import Callable, List

import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

import nlu
from data import snips, alexa
from data.entity import Utterance
from sbert import PseudoClassificationModel, get_pseudo_classifier_confidence, \
    _split_text_and_slots_into_tokens_and_tags
from ufd import Pipeline


def _populate_pr_auc_results(metrics, novelty, conf, output_folder,
                             feature_names: List[str] = None, utterances: List[Utterance] = None):
    os.makedirs(output_folder, exist_ok=True)

    pr_auc_results = {}

    baseline_prec = sum(novelty) / len(novelty)
    for m in metrics:
        prec, rec, threshold = precision_recall_curve(novelty, [-c[m] for c in conf])
        f1 = [0 if prec[i] + rec[i] == 0 else 2 * prec[i] * rec[i] / (prec[i] + rec[i]) for i in range(len(prec))]

        best_id = numpy.argmax(numpy.asarray(f1))
        pr_auc_results[m] = {
            'unseen.pr_auc_score': round(auc(rec, prec), 3),
            'unseen.optimal_f1': round(f1[best_id], 3),
            'unseen.threshold': -threshold[best_id],
            'unseen.prec': round(prec[best_id], 3),
            'unseen.rec': round(rec[best_id], 3),
        }
        unseen_f1 = f1[best_id]

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

        # Computing utterance-level stats for cluster-level
        if utterances is not None:
            truly_novel_features = set([feature_names[i] for i in range(len(novelty)) if novelty[i] == 1])
            predicted_novel_features = set(
                [feature_names[i] for i in range(len(novelty)) if conf[i][m] <= -threshold[best_id] + 1e-6])

            tp = len([u for u in utterances if
                      u.feature_name in truly_novel_features and u.feature_name in predicted_novel_features])
            prec_denom = len([u for u in utterances if u.feature_name in predicted_novel_features])
            rec_denom = len([u for u in utterances if u.feature_name in truly_novel_features])
            prec = 0 if prec_denom == 0 else tp / prec_denom
            rec = 0 if rec_denom == 0 else tp / rec_denom
            f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

            pr_auc_results[m].update({
                'unseen.utr.prec': round(prec, 3),
                'unseen.utr.rec': round(rec, 3),
                'unseen.utr.f1': round(f1, 3),
            })

        # Computing for negative
        tp = len([i for i in range(len(novelty)) if novelty[i] == 0 and conf[i][m] > -threshold[best_id] + 1e-6])
        prec_denom = len([c for c in conf if c[m] > -threshold[best_id] + 1e-6])
        rec_denom = len([n for n in novelty if n == 0])
        prec = 0 if prec_denom == 0 else tp / prec_denom
        rec = 0 if rec_denom == 0 else tp / rec_denom
        f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

        pr_auc_results[m].update({
            'seen.prec': round(prec, 3),
            'seen.rec': round(rec, 3),
            'seen.f1': round(f1, 3),
        })

        # Computing acc
        pr_auc_results[m].update({
            'seen+unseen.acc': round(sum([1 if (1 if conf[i][m] <= -threshold[best_id] + 1e-6 else 0) == novelty[i]
                                          else 0 for i in range(len(novelty))]) / len(novelty), 3),
            'seen+unseen.avg_f1': round((unseen_f1 + f1) / 2, 3)
        })

    with open(os.path.join(output_folder, 'stats.txt'), 'w') as f:
        f.write(json.dumps(pr_auc_results, indent=2))

    return pr_auc_results


def _update_conf(conf):
    conf['ner_tag_min x ic'] = conf['ic'] * conf['ner_tag_min']
    conf['ner_tag_min x pc'] = conf['pc'] * conf['ner_tag_min']
    conf['ner_tag_min x ic x pc'] = conf['ic'] * conf['ner_tag_min'] * conf['pc']
    conf['mean(ner_tag_min, ic)'] = (conf['ic'] + conf['ner_tag_min']) / 2
    conf['mean(ner_tag_min, pc)'] = (conf['ner_tag_min'] + conf['pc']) / 2
    conf['mean(ner_tag_min, ic, pc)'] = (conf['ic'] + conf['ner_tag_min'] + conf['pc']) / 3
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
                             os.path.join(output_folder, 'cluster_level'),
                             feature_names=[f for f in feature2novel], utterances=pipeline.test_utterances)
    _populate_pr_auc_results(conf.keys(), utterance_novelty, utterance_conf,
                             os.path.join(output_folder, 'utterance_level'))


def compute_dev_pr_auc_for_nlu_model_for_support_detection(pipeline: Pipeline, nlu_trained_model_path: str,
                                                           pseudo_classification_trained_model_path: str,
                                                           output_folder: str):
    nlu.load_finetuned(nlu_trained_model_path)

    # Cluster level
    feature2novel = {u.feature_name: 0 for u in pipeline.utterances if u.feature_name is not None and
                     u.feature_name.endswith('_TRAIN') and u.part_type == 'DEV'}
    feature2novel.update(
        {u.feature_name: 1 for u in pipeline.utterances if u.feature_name is not None and
         u.feature_name.endswith('_DEV') and u.part_type == 'DEV'})

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
    utterances = [u for u in pipeline.utterances if u.feature_name is not None and u.part_type == 'DEV']
    utterance_novelty = [1 if u.feature_name.endswith('_DEV') else 0 for u in utterances]
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
                             os.path.join(output_folder, 'cluster_level'),
                             feature_names=[f for f in feature2novel], utterances=utterances)
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
                individual_stats['conf']['ner_tag_min'][i],
                pc_conf[i].item() if pc is not None else None
            ))

        details[f] = {
            'conf': conf,
            'supported': supported,
            'utterances': utterances
        }

    def _quality(labels: List[int]):
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


def process_snips():
    _, inter_intent_data = snips.get_train_test_data(use_dev=True)
    p = Pipeline(inter_intent_data, dataset_name='inter_intent')

    compute_dev_pr_auc_for_nlu_model_for_support_detection(p,
                                                           './models/snips_nlu_exclude_unseen/inter_intent/nlu_model',
                                                           './reports/global/snips_SMC+US_PC_exclude_unseen/inter_intent/pc_trained_model',
                                                           './reports/nst/nlu_validation/snips_inter_intent_exclude_unseen/')
    compute_pr_auc_for_nlu_model_for_support_detection(p, './models/snips_nlu/inter_intent/nlu_model',
                                                       './reports/global/snips_SMC+US_PC/inter_intent/pc_trained_model',
                                                       './reports/nst/nlu_validation/snips_inter_intent/')
    # print(evaluate_nlu_model_for_support_detection(p, './models/snips_nlu/inter_intent/nlu_model',
    #                                                './reports/global/snips_SMC+US_PC/inter_intent/pc_trained_model',
    #                                                nst_callback=lambda conf: conf['ic x ner_tag_min x pc'] > 0.5,
    #                                                output_file='./reports/nst/nlu_validation/snips_inter_intent/cluster_stats_ic x ner_tag_min x pc.txt'
    #                                                ))


def process_alexa_fr():
    alexa_data = alexa.get_train_test_data()
    p = Pipeline(alexa_data, dataset_name='inter_intent')

    compute_dev_pr_auc_for_nlu_model_for_support_detection(p,
                                                           './models/alexa_fr_nlu_exclude_unseen/inter_intent/nlu_model',
                                                           './reports/global/alexa_fr/SMC+US_PC_exclude_unseen/inter_intent/pc_trained_model',
                                                           './reports/nst/nlu_validation/alexa_fr_exclude_unseen/')

    # compute_pr_auc_for_nlu_model_for_support_detection(p, './models/alexa_fr_nlu/inter_intent/nlu_model',
    #                                                    './reports/global/alexa_fr/SMC+US_PC_faiss_neu/inter_intent/pc_trained_model',
    #                                                    './reports/nst/nlu_validation/alexa_fr/')

    # print(evaluate_nlu_model_for_support_detection(p, './models/alexa_fr_nlu/inter_intent/nlu_model',
    #                                                './reports/global/alexa_fr/SMC+US_PC_faiss_neu/inter_intent/pc_trained_model',
    #                                                nst_callback=lambda conf: conf['mean(ner_tag_min, ic, pc)'] > 0.77,
    #                                                output_file='./reports/nst/nlu_validation/alexa_fr/cluster_stats_mean(ic,ner_tag_min,pc).txt'
    #                                                ))


if __name__ == '__main__':
    # process_snips()
    process_alexa_fr()
