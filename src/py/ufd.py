from typing import Union

import matplotlib.pyplot as plt
import umap
from numpy import ndarray

import sbert
from cluster import cop_kmeans, get_clustering_quality
from data import snips


def umap_plot(embeddings, labels, show_labels=False):
    embeddings = umap.UMAP().fit_transform(embeddings)
    u_labels = set(labels)
    for _, l in enumerate(u_labels):
        idx = [i for i, _ in enumerate(labels) if _ == l]
        plt.scatter([embeddings[i][0] for i in idx], [embeddings[i][1] for i in idx], label=l)
    if show_labels:
        plt.legend(loc='best')
    plt.show()


class Pipeline(object):

    def __init__(self, utterances: list[tuple[str, any, bool]]):  # (utterance, cluster_label, is_train)
        self.utterances = utterances
        self.cluster_label_2_index_map = dict((n, i) for i, n in enumerate(set([i for (_, i, _) in self.utterances])))

    def _get_embedding(self, utterances: Union[str, list[str]]) -> ndarray:
        if type(utterances) == list:
            return sbert.get_embeddings(utterances)
        else:
            return sbert.get_embeddings([utterances])

    def get_true_clusters(self):
        return [self.cluster_label_2_index_map[u[1]] for u in self.utterances]

    def get_pseudo_clusters(self, method='cop-kmeans', k=-1):
        train_clusters = [[]] * len(self.cluster_label_2_index_map)
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

        print('Getting embeddings')
        embeddings = self._get_embedding([u[0] for u in self.utterances])

        if method == 'cop-kmeans':
            if k <= 0:
                raise Exception('Invalid k={}'.format(k))

            print('Clustering: ', method)

            clusters, centers = cop_kmeans(dataset=embeddings, k=k, ml=ml, cl=cl)

            return clusters
        else:
            raise Exception('Method {} not supported'.format(method))

    def plot_2d(self):
        umap_plot(self._get_embedding([u[0] for u in self.utterances]), [u[1] for u in self.utterances])

    def find_tune(self):
        pass
        # TODO


if __name__ == '__main__':
    intent_data = snips.split_by_features_GetWeather()
    intent_map = dict((n, i) for i, n in enumerate(set([u['intent'] + '_' + u['cluster'] for u in intent_data])))

    snips_data = []
    for u in intent_data:
        snips_data.append((u['text'], u['intent'] + '_' + u['cluster'], False))
    p = Pipeline(snips_data)
    p.plot_2d()

    sbert_clusters = p.get_pseudo_clusters(k=len(p.cluster_label_2_index_map))
    print(get_clustering_quality(p.get_true_clusters(), sbert_clusters))

    pass
