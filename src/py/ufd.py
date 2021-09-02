from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import umap

import sbert
from cluster import cop_kmeans, get_clustering_quality


def umap_plot(embeddings, labels, sample_type=None, title=None, show_labels=False, plot_3d=False,
              output_file_path=None):
    if show_labels:
        plt.rcParams["figure.figsize"] = (10, 4)
    embeddings = umap.UMAP(n_components=3 if plot_3d else 2).fit_transform(embeddings)
    ax = plt.figure().add_subplot(projection='3d' if plot_3d else None)

    u_labels = set(labels)
    for _, l in enumerate(u_labels):
        idx = [i for i, _ in enumerate(labels) if _ == l]
        if plot_3d:
            if sample_type is None:
                ax.scatter([embeddings[i][0] for i in idx],
                           [embeddings[i][1] for i in idx],
                           [embeddings[i][2] for i in idx], label='{} ({})'.format(l, len(idx)), s=10)
            else:
                # sample_type is provided, draw them with different markers
                color = ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'TEST'],
                                   [embeddings[i][1] for i in idx if sample_type[i] == 'TEST'],
                                   [embeddings[i][2] for i in idx if sample_type[i] == 'TEST'],
                                   label='{} ({})'.format(l, len(idx)), s=10, marker='o').get_facecolor()[0]
                ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'DEV'],
                           [embeddings[i][1] for i in idx if sample_type[i] == 'DEV'],
                           [embeddings[i][2] for i in idx if sample_type[i] == 'DEV'],
                           s=10, color=color, marker='x')
                ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'TRAIN'],
                           [embeddings[i][1] for i in idx if sample_type[i] == 'TRAIN'],
                           [embeddings[i][2] for i in idx if sample_type[i] == 'TRAIN'],
                           s=10, color=color, marker='x')
        else:
            if sample_type is None:
                ax.scatter([embeddings[i][0] for i in idx],
                           [embeddings[i][1] for i in idx], label='{} ({})'.format(l, len(idx)), s=10)
            else:
                # sample_type is provided, draw them with different markers
                color = ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'TEST'],
                                   [embeddings[i][1] for i in idx if sample_type[i] == 'TEST'],
                                   label='{} ({})'.format(l, len(idx)), s=10, marker='o').get_facecolor()[0]
                ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'DEV'],
                           [embeddings[i][1] for i in idx if sample_type[i] == 'DEV'],
                           s=10, color=color, marker='x')
                ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'TRAIN'],
                           [embeddings[i][1] for i in idx if sample_type[i] == 'TRAIN'],
                           s=10, color=color, marker='x')

    if title is not None:
        plt.title(title)
    if show_labels:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()

    if output_file_path == None:
        plt.show()
    else:
        plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()


class Pipeline(object):

    def __init__(self, utterances: List[Tuple[str, any, str]]):  # (utterance, cluster_label, sample_type)
        self.use_dev = 'DEV' in [u[2] for u in utterances]
        self.utterances = []  # TRAIN + DEV, if DEV is not provided, use DEV = TEST instead.
        self.test_utterances = []
        for u in utterances:
            if u[2] in ['TRAIN', 'DEV']:
                self.utterances.append(u)
            elif u[2] == 'TEST':
                self.test_utterances.append(u)
                if not self.use_dev:
                    self.utterances.append(u)
            else:
                raise Exception('Invalid sample type')

        self.cluster_label_2_index_map = dict((n, i) for i, n in enumerate(set([j for (_, j, _) in self.utterances])))

        # TODO: use test_utterances

    def get_embeddings(self, utterances: List[str] = None):
        return sbert.get_embeddings([u[0] for u in self.utterances] if utterances is None else utterances)

    def get_true_clusters(self, including_train=True):
        if including_train:
            return [self.cluster_label_2_index_map[u[1]] for u in self.utterances]
        else:
            return [self.cluster_label_2_index_map[u[1]] for u in self.utterances if u[2] != 'TRAIN']

    # precomputed_embeddings is for train/dev only
    def get_pseudo_clusters(self, method='cop-kmeans', k=-1, precomputed_embeddings=None, including_train=True):
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

            print('Clustering:', method)

            clusters, centers = cop_kmeans(dataset=embeddings, k=k, ml=ml, cl=cl)

            if not including_train:
                clusters = [c for i, c in enumerate(clusters) if self.utterances[i][2] != 'TRAIN']
            return clusters
        else:
            raise Exception('Method {} not supported'.format(method))

    # precomputed_embeddings is for train/dev only
    def plot(self, title=None, show_labels=True, show_sample_type=True, precomputed_embeddings=None, plot_3d=False,
             output_file_path=None):
        train_dev_embeddings = precomputed_embeddings if precomputed_embeddings is not None else self.get_embeddings()

        # Compute test_embeddings only when use_dev is True, otherwise it has been already included in train_dev_embeddings
        test_embeddings = self.get_embeddings([u[0] for u in self.test_utterances]) if self.use_dev \
            else numpy.ndarray((0, train_dev_embeddings.shape[1]))

        embeddings = numpy.concatenate([train_dev_embeddings, test_embeddings])

        labels = [u[1] for u in self.utterances] + ([u[1] for u in self.test_utterances] if self.use_dev else [])
        sample_type = [u[2] for u in self.utterances] + ([u[2] for u in self.test_utterances] if self.use_dev else [])

        umap_plot(embeddings, labels, sample_type if show_sample_type else None,
                  title=title, show_labels=show_labels, plot_3d=plot_3d, output_file_path=output_file_path)

    def find_tune_pseudo_classification(self, k=None, precomputed_embeddings=None):
        pseudo_clusters = self.get_pseudo_clusters(k=k if k is not None else len(self.cluster_label_2_index_map),
                                                   precomputed_embeddings=precomputed_embeddings)
        print('Pseudo-cluster quality:',
              get_clustering_quality(self.get_true_clusters(), pseudo_clusters))
        sbert.fine_tune_classification([u[0] for u in self.utterances], pseudo_clusters)

    def find_tune_utterance_similarity(self):
        cluster_indices = [u[1] if u[2] == 'TRAIN' else None for u in self.utterances]
        sbert.fine_tune_utterance_similarity([u[0] for u in self.utterances], cluster_indices)
