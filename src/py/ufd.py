from typing import Union

from numpy import ndarray
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

import sbert
from cluster import cop_kmeans
from data import snips


class Pipeline(object):

    def __init__(self, utterances: list[tuple[str, int, bool]]):  # (utterance, cluster_index, is_train)
        self.utterances = utterances

    def _get_embedding(self, utterances: Union[str, list[str]]) -> ndarray:
        if type(utterances) == list:
            return sbert.get_embeddings(utterances)
        else:
            return sbert.get_embeddings([utterances])

    def get_pseudo_clusters(self, method='cop-kmeans', k=-1):
        n_clusters = 1 + max([i for (_, i, _) in self.utterances])
        train_clusters = [[] for _ in range(n_clusters)]
        for i, (_, j, t) in enumerate(self.utterances):
            if t:
                train_clusters[j].append(i)

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
        embeddings = self._get_embedding([u[0] for u, _, _ in self.utterances])

        if method == 'cop-kmeans':
            if k <= 0:
                raise Exception('Invalid k={}'.format(k))

            print('Clustering: ', method)

            clusters, centers = cop_kmeans(dataset=embeddings, k=k, ml=ml, cl=cl)

            return clusters
        else:
            raise Exception('Method {} not supported'.format(method))


if __name__ == '__main__':
    snips_data = snips.get_all_utterances_and_labels()

    true_clusters = [i for _, i, _ in snips_data]
    p = Pipeline(snips_data)

    sbert_clusters = p.get_pseudo_clusters(k=7)
    print("NMI: {}".format(normalized_mutual_info_score(true_clusters, sbert_clusters)))
    print("ARI: {}".format(adjusted_rand_score(true_clusters, sbert_clusters)))
