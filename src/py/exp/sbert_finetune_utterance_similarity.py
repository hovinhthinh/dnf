import os

import sbert
from cluster import get_clustering_quality
from data import snips
from data.snips import print_train_dev_test_stats
from ufd import Pipeline

output_file_path = './reports/snips_finetune_utterance_similarity'

intra_intent_data, inter_intent_data = snips.get_train_test_data()

# Processing intra-intents
for name, intent_data in intra_intent_data:
    print('======== Intra-intent:', name, '========')
    print_train_dev_test_stats(intent_data)

    p = Pipeline(intent_data)
    if output_file_path is not None:
        os.makedirs(output_file_path + '/' + name, exist_ok=True)

    sbert.load()
    embeddings = p.get_embeddings()
    p.plot(title=name, precomputed_embeddings=embeddings, plot_3d=False,
           output_file_path=output_file_path + '/' + name + '/0.pdf' if output_file_path is not None else None)
    print('Quality of clustering unseen before fine-tuning:',
          get_clustering_quality(p.get_true_clusters(including_train=False),
                                 p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                       precomputed_embeddings=embeddings, including_train=False)))
    for it in range(10):
        print('Iter: #{}'.format(it + 1))
        p.find_tune_utterance_similarity()
        embeddings = p.get_embeddings()
        p.plot(title=name, precomputed_embeddings=embeddings, plot_3d=False,
               output_file_path=output_file_path + '/' + name + '/{}.pdf'.format(it + 1)
               if output_file_path is not None else None)

    print('Quality of clustering unseen after fine-tuning:',
          get_clustering_quality(p.get_true_clusters(including_train=False),
                                 p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                       precomputed_embeddings=embeddings, including_train=False)))

# Processing inter-intents
inter_intent_data = [u for u in inter_intent_data if not u[1].startswith('BookRestaurant_')]

print('======== Inter-intent ========')
train_size = len([u for u in inter_intent_data if u[2]])
print_train_dev_test_stats(intent_data)

p = Pipeline(inter_intent_data)
if output_file_path is not None:
    os.makedirs(output_file_path + '/inter-intent', exist_ok=True)

sbert.load()
embeddings = p.get_embeddings()
p.plot(title='inter-intent', precomputed_embeddings=embeddings, plot_3d=False,
       output_file_path=output_file_path + '/inter-intent' + '/0.pdf' if output_file_path is not None else None)
print('Quality of clustering unseen before fine-tuning:',
      get_clustering_quality(p.get_true_clusters(including_train=False),
                             p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                   precomputed_embeddings=embeddings, including_train=False)))
for it in range(10):
    print('Iter: #{}'.format(it + 1))
    p.find_tune_utterance_similarity()
    embeddings = p.get_embeddings()
    p.plot(title='inter-intent', precomputed_embeddings=embeddings, plot_3d=False,
           output_file_path=output_file_path + '/inter-intent' + '/{}.pdf'.format(it + 1)
           if output_file_path is not None else None)

print('Quality of clustering unseen after fine-tuning:',
      get_clustering_quality(p.get_true_clusters(including_train=False),
                             p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map),
                                                   precomputed_embeddings=embeddings, including_train=False)))

# Apply back to intra-intent
print('======== Apply back to intra-intent ========')
for name, intent_data in intra_intent_data:
    print('======== Intent:', name, '========')
    p = Pipeline(intent_data)
    print('Quality of clustering unseen before fine-tuning:',
          get_clustering_quality(p.get_true_clusters(including_train=False),
                                 p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map), including_train=False)))
