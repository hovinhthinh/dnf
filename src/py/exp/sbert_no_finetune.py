import os

import sbert
from cluster import get_clustering_quality
from data import snips
from ufd import Pipeline

output_file_path = './reports/snips_no_finetune'
if output_file_path is not None:
    os.makedirs(output_file_path, exist_ok=True)

sbert.load('sentence-transformers/paraphrase-mpnet-base-v2')

intent_data = [
    (snips.split_by_features_GetWeather(), '/intra_intent_GetWeather.pdf', 'intra_intent_GetWeather'),
    (snips.split_by_features_AddToPlaylist(), '/intra_intent_AddToPlaylist.pdf', 'intra_intent_AddToPlaylist'),
    (snips.split_by_features_RateBook(), '/intra_intent_RateBook.pdf', 'intra_intent_RateBook'),
    (snips.split_by_features_BookRestaurant(), '/intra_intent_BookRestaurant.pdf', 'intra_intent_BookRestaurant'),
    (snips.split_by_features_PlayMusic(), '/intra_intent_PlayMusic.pdf', 'intra_intent_PlayMusic'),
    (snips.split_by_features_SearchCreativeWork(), '/intra_intent_SearchCreativeWork.pdf',
     'intra_intent_SearchCreativeWork'),
    (snips.split_by_features_SearchScreeningEvent(), '/intra_intent_SearchScreeningEvent.pdf',
     'intra_intent_SearchScreeningEvent'),
]

# inter-intent
inter_intent_data = []
for data, output, title in intent_data:
    for u in data:
        inter_intent_data.append((u['text'], u['intent'], False))
p = Pipeline(inter_intent_data)
embeddings = p.get_embeddings()
p.plot(precomputed_embeddings=embeddings, show_labels=True,
       output_file_path=output_file_path + '/inter_intent.pdf', title='inter intent')

sbert_clusters = p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map), precomputed_embeddings=embeddings)
print('Inter-intent clustering-metrics:', get_clustering_quality(p.get_true_clusters(), sbert_clusters))

# intra-intent
for data, output, title in intent_data:
    snips_data = []
    for u in data:
        snips_data.append((u['text'], u['cluster'], False))
    p = Pipeline(snips_data)
    embeddings = p.get_embeddings()
    p.plot(precomputed_embeddings=embeddings, show_labels=True, output_file_path=output_file_path + output, title=title)

    sbert_clusters = p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map), precomputed_embeddings=embeddings)
    print('Intra-intent clustering-metrics -', title, ':',
          get_clustering_quality(p.get_true_clusters(), sbert_clusters))
