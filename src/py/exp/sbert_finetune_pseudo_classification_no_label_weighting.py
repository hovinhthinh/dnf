# 60% clusters are known, in each known cluster, we have labels of 60% utterances.
import json
import os
import random

import sbert
from cluster import get_clustering_quality
from data import snips
from ufd import Pipeline

output_file_path = './reports/snips_finetune_pseudo_classification/no_label_weighting'

generate_data = False

if generate_data:
    intent_data = [
        (snips.split_by_features_GetWeather(), 'GetWeather'),
        (snips.split_by_features_AddToPlaylist(), 'AddToPlaylist'),
        (snips.split_by_features_RateBook(), 'RateBook'),
        (snips.split_by_features_BookRestaurant(), 'BookRestaurant'),
        (snips.split_by_features_PlayMusic(), 'PlayMusic'),
        (snips.split_by_features_SearchCreativeWork(), 'SearchCreativeWork'),
        (snips.split_by_features_SearchScreeningEvent(), 'SearchScreeningEvent'),
    ]

    inter_intent_data = []
    intra_intent_data = []
    # Prepare data
    for data, name in intent_data:
        clusters = list(set([u['cluster'] for u in data]))
        # Filter clusters with small size
        clusters = [c for c in clusters if sum([1 for u in data if u['cluster'] == c]) >= 40]

        # Random train clusters
        random.shuffle(clusters)
        train_clusters = clusters[0:int(len(clusters) * 0.6)]
        print('Train/Test clusters:', train_clusters, clusters[int(len(clusters) * 0.6):])

        # Split into train/test utterances
        splitted_data = []
        for c in clusters:
            cluster_data = [u for u in data if u['cluster'] == c]

            if c in train_clusters:
                random.shuffle(cluster_data)
                train_size = int(len(cluster_data) * 0.6)
                splitted_data.extend([(u['text'], c + '_T', True) for u in cluster_data[0:train_size]])
                splitted_data.extend([(u['text'], c + '_T', False) for u in cluster_data[train_size:]])
            else:
                splitted_data.extend([(u['text'], c, False) for u in cluster_data])

        intra_intent_data.append((name, splitted_data))
        inter_intent_data.extend(
            [(text, data[0]['intent'] + '_' + cluster, is_train) for text, cluster, is_train in splitted_data])

    with open(output_file_path + '/data.json', 'w') as f:
        f.write(json.dumps((intra_intent_data, inter_intent_data)))
else:
    intra_intent_data, inter_intent_data = json.loads(open(output_file_path + '/data.json').read())

# Processing intra-intents
for name, intent_data in intra_intent_data:
    if name == 'BookRestaurant':
        continue
    print('======== Intra-intent:', name, '========')
    train_size = len([u for u in intent_data if u[2]])
    print('Training utterances: {}/{} ({:.1f}%)'.format(train_size, len(intent_data),
                                                        100 * train_size / len(intent_data)))
    p = Pipeline(intent_data)
    if output_file_path is not None:
        os.makedirs(output_file_path + '/' + name, exist_ok=True)

    sbert.load('./models/sentence-transformers.paraphrase-mpnet-base-v2')
    embeddings = p.get_embeddings()
    p.plot(title=name, show_labels=True, precomputed_embeddings=embeddings, plot_3d=False,
           output_file_path=output_file_path + '/' + name + '/0.pdf' if output_file_path is not None else None)
    for it in range(10):
        print('Iter: #{}'.format(it + 1))
        p.find_tune_pseudo_classification(precomputed_embeddings=embeddings)
        embeddings = p.get_embeddings()
        p.plot(title=name, show_labels=True, precomputed_embeddings=embeddings, plot_3d=False,
               output_file_path=output_file_path + '/' + name + '/{}.pdf'.format(it + 1)
               if output_file_path is not None else None)

    predicted_clusters = p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                               precomputed_embeddings=embeddings)
    print('Clustering quality after fine-tuning:',
          get_clustering_quality(p.get_true_clusters(), predicted_clusters))

# Processing inter-intents
inter_intent_data = [u for u in inter_intent_data if not u[1].startswith('BookRestaurant_')]

print('======== Inter-intent ========')
train_size = len([u for u in inter_intent_data if u[2]])
print('Training utterances: {}/{} ({:.1f}%)'.format(train_size, len(inter_intent_data),
                                                    100 * train_size / len(inter_intent_data)))
p = Pipeline(inter_intent_data)
if output_file_path is not None:
    os.makedirs(output_file_path + '/inter-intent', exist_ok=True)

sbert.load('./models/sentence-transformers.paraphrase-mpnet-base-v2')
embeddings = p.get_embeddings()
p.plot(title='inter-intent', show_labels=True, precomputed_embeddings=embeddings, plot_3d=False,
       output_file_path=output_file_path + '/inter-intent' + '/0.pdf' if output_file_path is not None else None)
for it in range(10):
    print('Iter: #{}'.format(it + 1))
    p.find_tune_pseudo_classification(precomputed_embeddings=embeddings)
    embeddings = p.get_embeddings()
    p.plot(title='inter-intent', show_labels=True, precomputed_embeddings=embeddings, plot_3d=False,
           output_file_path=output_file_path + '/inter-intent' + '/{}.pdf'.format(it + 1)
           if output_file_path is not None else None)

predicted_clusters = p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                           precomputed_embeddings=embeddings)
print('Clustering quality after fine-tuning:',
      get_clustering_quality(p.get_true_clusters(), predicted_clusters))
