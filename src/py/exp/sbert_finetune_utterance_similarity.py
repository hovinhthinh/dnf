# 60% clusters are known, in each known cluster, we have labels of 60% utterances.
import os

import sbert
from cluster import get_clustering_quality
from data import snips
from ufd import Pipeline

output_file_path = './reports/snips_finetune_utterance_similarity'

intra_intent_data, inter_intent_data = snips.get_train_test_data()

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
    print('Clustering quality before fine-tuning:',
          get_clustering_quality(p.get_true_clusters(test_only=True),
                                 p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                       precomputed_embeddings=embeddings, test_only=True)))
    for it in range(10):
        print('Iter: #{}'.format(it + 1))
        p.find_tune_utterance_similarity()
        embeddings = p.get_embeddings()
        p.plot(title=name, show_labels=True, precomputed_embeddings=embeddings, plot_3d=False,
               output_file_path=output_file_path + '/' + name + '/{}.pdf'.format(it + 1)
               if output_file_path is not None else None)

    print('Clustering quality after fine-tuning:',
          get_clustering_quality(p.get_true_clusters(test_only=True),
                                 p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                       precomputed_embeddings=embeddings, test_only=True)))

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
print('Clustering quality before fine-tuning:',
      get_clustering_quality(p.get_true_clusters(test_only=True),
                             p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                   precomputed_embeddings=embeddings, test_only=True)))
for it in range(10):
    print('Iter: #{}'.format(it + 1))
    p.find_tune_utterance_similarity()
    embeddings = p.get_embeddings()
    p.plot(title='inter-intent', show_labels=True, precomputed_embeddings=embeddings, plot_3d=False,
           output_file_path=output_file_path + '/inter-intent' + '/{}.pdf'.format(it + 1)
           if output_file_path is not None else None)

print('Clustering quality after fine-tuning:',
      get_clustering_quality(p.get_true_clusters(test_only=True),
                             p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                   precomputed_embeddings=embeddings, test_only=True)))

# Apply back to intra-intent
print('======== Apply back to intra-intent ========')
for name, intent_data in intra_intent_data:
    if name == 'BookRestaurant':
        continue
    print('======== Intent:', name, '========')
    p = Pipeline(intent_data)
    print('Clustering quality after fine-tuning:',
          get_clustering_quality(p.get_true_clusters(test_only=True),
                                 p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map), test_only=True)))
