import matplotlib.pyplot as plt
import umap

import sbert
from cluster import cop_kmeans, get_clustering_quality
from data import snips


def umap_plot(embeddings, labels, show_labels=False):
    if show_labels:
        plt.rcParams["figure.figsize"] = (10, 4)
    embeddings = umap.UMAP().fit_transform(embeddings)
    u_labels = set(labels)
    for _, l in enumerate(u_labels):
        idx = [i for i, _ in enumerate(labels) if _ == l]
        plt.scatter([embeddings[i][0] for i in idx], [embeddings[i][1] for i in idx], label=l)
    if show_labels:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


class Pipeline(object):

    def __init__(self, utterances: list[tuple[str, any, bool]]):  # (utterance, cluster_label, is_train)
        self.utterances = utterances
        self.cluster_label_2_index_map = dict((n, i) for i, n in enumerate(set([j for (_, j, _) in self.utterances])))

    def get_embeddings(self):
        return sbert.get_embeddings([u[0] for u in self.utterances])

    def get_true_clusters(self):
        return [self.cluster_label_2_index_map[u[1]] for u in self.utterances]

    def get_pseudo_clusters(self, method='cop-kmeans', k=-1, precomputed_embeddings=None):
        train_clusters = [[] for _ in range(len(self.cluster_label_2_index_map))]
        for i, (_, j, t) in enumerate(self.utterances):
            if t:
                train_clusters[self.cluster_label_2_index_map[j]].append(i)

        # Constraints
        ml = []
        cl = []
        for i in range(len(train_clusters)):
            if len(train_clusters[i]) == 0:
                continue

            for j in range(len(train_clusters[i]) - 1):
                ml.append((train_clusters[i][j], train_clusters[i][j + 1]))
            for j in range(i + 1, len(train_clusters)):
                if len(train_clusters[j]) == 0:
                    continue
                cl.append((train_clusters[i][0], train_clusters[j][0]))

        embeddings = precomputed_embeddings if precomputed_embeddings is not None else self.get_embeddings()

        if method == 'cop-kmeans':
            if k <= 0:
                raise Exception('Invalid k={}'.format(k))

            print('Clustering: ', method)

            clusters, centers = cop_kmeans(dataset=embeddings, k=k, ml=ml, cl=cl)

            return clusters
        else:
            raise Exception('Method {} not supported'.format(method))

    def plot_2d(self, show_labels=False, precomputed_embeddings=None):
        umap_plot(precomputed_embeddings if precomputed_embeddings is not None else self.get_embeddings(),
                  [u[1] for u in self.utterances], show_labels=show_labels)

    def find_tune_pseudo_classification(self, k=None, precomputed_embeddings=None):
        pseudo_clusters = p.get_pseudo_clusters(k=k if k is not None else len(self.cluster_label_2_index_map),
                                                precomputed_embeddings=precomputed_embeddings)
        sbert.fine_tune_classification([u[0] for u in self.utterances], pseudo_clusters)


if __name__ == '__main__':
    intent_data = snips.split_by_features_GetWeather()
    # The commented code below is to clustering only by intents
    # intent_data.extend(snips.split_by_features_AddToPlaylist())
    # intent_data.extend(snips.split_by_features_RateBook())
    # intent_data.extend(snips.split_by_features_BookRestaurant())
    # intent_data.extend(snips.split_by_features_PlayMusic())
    # intent_data.extend(snips.split_by_features_SearchCreativeWork())
    # intent_data.extend(snips.split_by_features_SearchScreeningEvent())
    # for u in intent_data:
    #     u['cluster'] = 'UNK'

    # intent_data = snips.split_by_features_AddToPlaylist()
    # intent_data = snips.split_by_features_RateBook()
    # intent_data = snips.split_by_features_BookRestaurant()
    # intent_data = snips.split_by_features_PlayMusic()
    # intent_data = snips.split_by_features_SearchCreativeWork()
    # intent_data = snips.split_by_features_SearchScreeningEvent()

    intent_map = dict((n, i) for i, n in enumerate(set([u['intent'] + '_' + u['cluster'] for u in intent_data])))

    sbert.load('sentence-transformers/paraphrase-mpnet-base-v2')

    # Prepare data
    snips_data = []
    for u in intent_data:
        is_train = u['cluster'] in ['GetCurrentWeatherInALocation', 'GetWeatherInCurrentPositionAtATimeRange']
        snips_data.append((u['text'], u['intent'] + '_' + u['cluster'] + ('_T' if is_train else ''), is_train))

    p = Pipeline(snips_data)

    # Pseudo clustering
    embeddings = p.get_embeddings()
    p.plot_2d(show_labels=True, precomputed_embeddings=embeddings)
    for iter in range(10):
        print('Iter: #{}'.format(iter))
        p.find_tune_pseudo_classification(precomputed_embeddings=embeddings)
        embeddings = p.get_embeddings()
        p.plot_2d(show_labels=True, precomputed_embeddings=embeddings)

    sbert_clusters = p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map), precomputed_embeddings=embeddings)
    print(get_clustering_quality(p.get_true_clusters(), sbert_clusters))

    pass
