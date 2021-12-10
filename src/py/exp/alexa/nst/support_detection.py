import json
import statistics

import nlu
from data import alexa
from nst import _update_conf
from sbert import PseudoClassificationModel, get_pseudo_classifier_confidence
from ufd import Pipeline

alexa_data = alexa.get_train_test_data()
p = Pipeline(alexa_data, dataset_name='inter_intent')

# Load NLU model
nlu.load_finetuned('./models/alexa_fr_nlu/inter_intent/nlu_model')
# Load Pseudo classifier
pc = PseudoClassificationModel.load_model('./reports/global/alexa_fr/SMC+US_PC_faiss_neu/inter_intent/pc_trained_model')
# Load thresholding stats
metric2threshold = {
    k: v['unseen.threshold'] for k, v in
    json.loads(open('./reports/nst/nlu_validation/alexa_fr/cluster_level/stats.txt').read()).items()
}

# Evaluate predicted clusters
predicted_clusters = json.loads(
    open('./reports/global/alexa_fr/SMC+US_PC_elbow_faiss_neu/inter_intent/2_PC_test.predicted_clusters.stats.json',
         'r').read())

clusters = predicted_clusters.pop('clusters')

with open('reports/nst/prediction/alexa_fr_stats-kmeans_elbow.txt', 'w') as f:
    clusters_with_confidences = []
    for c in clusters:
        test_ids = c.pop('test_utterance_ids')
        nlu_quality = p.get_nlu_test_quality(test_ids)
        pc_conf = get_pseudo_classifier_confidence([p.test_utterances[i].text for i in test_ids], pc)

        conf = nlu_quality.pop('conf')
        conf['pc'] = statistics.mean(pc_conf).item()
        _update_conf(conf)

        cluster_quality = c.pop('quality')

        cc = {
            'stats': c,
            'quality': cluster_quality,
            'conf': conf,
        }
        print(cc)
        clusters_with_confidences.append(cc)

    # Now measuring prediction quality
    metric2threshold['all'] = None

    predicted_quality = {}
    total_novel_clusters = len(set([u.feature_name for u in p.test_utterances if u.feature_name.endswith('_TEST')]))
    for m, threshold in metric2threshold.items():
        qualified_clusters = [c for c in clusters_with_confidences if
                              threshold is None or c['conf'][m] <= threshold + 1e-6]
        tp = len(set([c['stats']['main_feature'] for c in qualified_clusters if
                      c['stats']['main_feature'].endswith('_TEST')]))
        prec = tp / len(qualified_clusters)
        rec = tp / total_novel_clusters
        f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        predicted_quality[m] = {
            'threshold': threshold,
            'n_predicted_novel': len(qualified_clusters),
            'avg_novelty': round(statistics.mean([c['quality']['novelty'] for c in qualified_clusters]), 3),
            'prec': round(prec, 3),
            'rec': round(rec, 3),
            'f1': round(f1, 3),
        }

    output = {
        'metrics': predicted_quality,
        'clusters': clusters_with_confidences
    }

    f.write(f'{json.dumps(output, indent=2)}\n')
