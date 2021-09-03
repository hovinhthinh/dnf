import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import umap
from sklearn.cluster import KMeans

import sbert
from cluster import cop_kmeans, get_clustering_quality


def umap_plot(embeddings, labels, sample_type=None, title=None, show_labels=False, plot_3d=False,
              label_plotting_order=None, output_file_path=None):
    if show_labels:
        plt.rcParams["figure.figsize"] = (10, 4)
    embeddings = umap.UMAP(n_components=3 if plot_3d else 2).fit_transform(embeddings)
    ax = plt.figure().add_subplot(projection='3d' if plot_3d else None)

    u_labels = set(labels)
    if label_plotting_order is None:
        label_plotting_order = [(l, None) for l in u_labels]

    for l, lc in label_plotting_order:
        if l not in u_labels:
            continue
        idx = [i for i, _ in enumerate(labels) if _ == l]
        if plot_3d:
            if sample_type is None:
                ax.scatter([embeddings[i][0] for i in idx],
                           [embeddings[i][1] for i in idx],
                           [embeddings[i][2] for i in idx], label='{} ({})'.format(l, len(idx)), s=10, color=lc)
            else:
                # sample_type is provided, draw them with different markers
                color = ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'TEST'],
                                   [embeddings[i][1] for i in idx if sample_type[i] == 'TEST'],
                                   [embeddings[i][2] for i in idx if sample_type[i] == 'TEST'],
                                   label='{} ({})'.format(l, len(idx)), s=10, marker='o', color=lc).get_facecolor()[0]
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
                           [embeddings[i][1] for i in idx], label='{} ({})'.format(l, len(idx)), s=10, color=lc)
            else:
                # sample_type is provided, draw them with different markers
                color = ax.scatter([embeddings[i][0] for i in idx if sample_type[i] == 'TEST'],
                                   [embeddings[i][1] for i in idx if sample_type[i] == 'TEST'],
                                   label='{} ({})'.format(l, len(idx)), s=10, marker='o', color=lc).get_facecolor()[0]
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

    def __init__(self, utterances: List[Tuple[str, any, str]],  # (utterance, cluster_label, sample_type)
                 dataset_name=None):
        self.dataset_name = dataset_name
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

        # Label plotting order
        self.label_plotting_order = []
        for u in [u[1] for u in utterances if u[2] == 'TRAIN'] \
                 + [u[1] for u in utterances if u[2] == 'DEV'] \
                 + [u[1] for u in utterances if u[2] == 'TEST']:
            if u in self.label_plotting_order:
                continue
            self.label_plotting_order.append(u)

        # pseudo-scatter for getting colors.
        ax = plt.figure().add_subplot()
        self.label_plotting_order = [(l, ax.scatter([], []).get_facecolor()[0]) for l in self.label_plotting_order]
        plt.close()

    def get_embeddings(self, utterances: List[str] = None):
        return sbert.get_embeddings([u[0] for u in self.utterances] if utterances is None else utterances)

    def get_test_embeddings(self):
        return sbert.get_embeddings([u[0] for u in self.test_utterances])

    def get_true_clusters(self, including_train=True):
        if including_train:
            return [self.cluster_label_2_index_map[u[1]] for u in self.utterances]
        else:
            return [self.cluster_label_2_index_map[u[1]] for u in self.utterances if u[2] != 'TRAIN']

    # precomputed_embeddings is for train/dev only
    def get_pseudo_clusters(self, method='cop-kmeans', k=-1, precomputed_embeddings=None, including_train=True):
        train_clusters = [[] for _ in range(len(self.cluster_label_2_index_map))]
        for i, (_, j, t) in enumerate(self.utterances):
            if t == 'TRAIN':
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
    def plot(self, show_train_dev_only=False, show_test_only=False, show_labels=True, show_sample_type=True,
             precomputed_embeddings=None, precomputed_test_embeddings=None, plot_3d=False,
             output_file_path=None):
        if show_train_dev_only:
            embeddings = precomputed_embeddings if precomputed_embeddings is not None else self.get_embeddings()
            labels = [u[1] for u in self.utterances]
            sample_type = [u[2] for u in self.utterances]
            umap_plot(embeddings, labels, sample_type if show_sample_type else None,
                      title=self.dataset_name, show_labels=show_labels, plot_3d=plot_3d,
                      label_plotting_order=self.label_plotting_order, output_file_path=output_file_path)
        elif show_test_only:
            if self.use_dev or precomputed_embeddings is None:
                test_embeddings = precomputed_test_embeddings if precomputed_test_embeddings is not None \
                    else self.get_test_embeddings()
            else:
                indices = [i for i, u in enumerate(self.utterances) if u[2] == 'TEST']
                test_embeddings = precomputed_embeddings[indices]

            test_labels = [u[1] for u in self.test_utterances]

            umap_plot(test_embeddings, test_labels, title=self.dataset_name, show_labels=show_labels, plot_3d=plot_3d,
                      label_plotting_order=self.label_plotting_order, output_file_path=output_file_path)
        else:
            train_dev_embeddings = precomputed_embeddings if precomputed_embeddings is not None else self.get_embeddings()

            # Compute test_embeddings only when use_dev is True, otherwise it has been already included in train_dev_embeddings
            if self.use_dev:
                test_embeddings = precomputed_test_embeddings if precomputed_test_embeddings is not None \
                    else self.get_test_embeddings()
            else:
                test_embeddings = numpy.ndarray((0, train_dev_embeddings.shape[1]))

            embeddings = numpy.concatenate([train_dev_embeddings, test_embeddings])

            labels = [u[1] for u in self.utterances] + ([u[1] for u in self.test_utterances] if self.use_dev else [])
            sample_type = [u[2] for u in self.utterances] + (
                [u[2] for u in self.test_utterances] if self.use_dev else [])

            umap_plot(embeddings, labels, sample_type if show_sample_type else None,
                      title=self.dataset_name, show_labels=show_labels, plot_3d=plot_3d,
                      label_plotting_order=self.label_plotting_order, output_file_path=output_file_path)

    def find_tune_pseudo_classification(self, k=None, precomputed_embeddings=None):
        pseudo_clusters = self.get_pseudo_clusters(k=k if k is not None else len(self.cluster_label_2_index_map),
                                                   precomputed_embeddings=precomputed_embeddings)
        print('Pseudo-cluster quality:',
              get_clustering_quality(self.get_true_clusters(), pseudo_clusters))
        sbert.fine_tune_classification([u[0] for u in self.utterances], pseudo_clusters)

    def find_tune_utterance_similarity(self):
        cluster_indices = [u[1] if u[2] == 'TRAIN' else None for u in self.utterances]
        sbert.fine_tune_utterance_similarity([u[0] for u in self.utterances], cluster_indices)

    def get_test_quality(self, precomputed_test_embedding=None):
        test_cluster_label_2_index_map = dict(
            (n, i) for i, n in enumerate(set([j for (_, j, _) in self.test_utterances])))
        test_true_clusters = [test_cluster_label_2_index_map[u[1]] for u in self.test_utterances]

        test_embeddings = precomputed_test_embedding if precomputed_test_embedding is not None \
            else self.get_test_embeddings()
        test_predicted_clusters = KMeans(n_clusters=len(test_cluster_label_2_index_map)).fit(test_embeddings).labels_

        return get_clustering_quality(test_true_clusters, test_predicted_clusters)

    def run(self, report_folder=None, steps=[
        'test-no-finetune',
        'finetune-utterance-similarity'
        'test-utterance-similarity',
        'finetune-pseudo-classification',
        'test-pseudo-classification',
    ]):
        for s in steps:
            if s not in [
                'test-no-finetune',
                'finetune-utterance-similarity'
                'test-utterance-similarity',
                'finetune-pseudo-classification',
                'test-pseudo-classification',
            ]:
                raise Exception('Invalid step name:', s)

        sbert.load()

        stats_file = None
        if report_folder is not None:
            os.makedirs(report_folder, exist_ok=True)
            stats_file = open(os.path.join(report_folder, 'stats.txt'), 'w')

        if 'test-no-finetune' in steps:
            folder = None
            if report_folder is not None:
                folder = os.path.join(report_folder, 'no-finetune')
                os.makedirs(folder, exist_ok=True)
            train_dev_embeddings = self.get_embeddings()
            test_embeddings = self.get_test_embeddings()
            self.plot(show_train_dev_only=True, plot_3d=False,
                      precomputed_embeddings=train_dev_embeddings,
                      output_file_path=os.path.join(folder, 'train-dev.pdf') if folder is not None else None)
            self.plot(plot_3d=False,
                      precomputed_embeddings=train_dev_embeddings, precomputed_test_embeddings=test_embeddings,
                      output_file_path=os.path.join(folder, 'train-dev-test.pdf') if folder is not None else None)

            test_quality = self.get_test_quality(precomputed_test_embedding=test_embeddings)
            print('No-finetune test quality:', test_quality)
            if stats_file is not None:
                stats_file.write('No-finetune test quality: {}\n'.format(test_quality))

        if 'finetune-utterance-similarity' in steps:
            pass
        if 'test-utterance-similarity' in steps:
            pass
        if 'finetune-pseudo-classification' in steps:
            pass
        if 'test-pseudo-classification' in steps:
            pass

        if stats_file is not None:
            stats_file.close()
