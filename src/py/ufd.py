import os
from math import pow
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

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

    def __init__(self, utterances: List[Tuple[str, any, str, dict]],  # (utterance, cluster_label, sample_type, slots)
                 dataset_name=None, normalize_embeddings=False):
        self.dataset_name = dataset_name
        self.normalize_embeddings = normalize_embeddings
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

        self.cluster_label_2_index_map = dict((n, i) for i, n in enumerate(set([u[1] for u in self.utterances])))

        self.embeddings = None
        self.test_embeddings = None

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

    def update_embeddings(self):
        self.embeddings = sbert.get_embeddings([u[0] for u in self.utterances])

    def update_test_embeddings(self, reuse_from_train_dev=True):
        if self.use_dev or self.embeddings is None or not reuse_from_train_dev:
            self.test_embeddings = sbert.get_embeddings([u[0] for u in self.test_utterances])
        else:
            indices = [i for i, u in enumerate(self.utterances) if u[2] == 'TEST']
            self.test_embeddings = self.embeddings[indices]

    def get_true_clusters(self, including_train=True):
        if including_train:
            return [self.cluster_label_2_index_map[u[1]] for u in self.utterances]
        else:
            return [self.cluster_label_2_index_map[u[1]] for u in self.utterances if u[2] != 'TRAIN']

    # Returns pseudo clusters and assignment confidences
    def get_pseudo_clusters(self, method='cop-kmeans', k=-1, including_train=True):
        train_clusters = [[] for _ in range(len(self.cluster_label_2_index_map))]
        for i, u in enumerate(self.utterances):
            if u[2] == 'TRAIN':
                train_clusters[self.cluster_label_2_index_map[u[1]]].append(i)

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

        if method == 'cop-kmeans':
            if k <= 0:
                raise Exception('Invalid k={}'.format(k))

            print('Clustering:', method)

            if self.normalize_embeddings:
                scaler = StandardScaler()
                scaler.fit(self.embeddings)
                embeddings = scaler.transform(self.embeddings)
            else:
                embeddings = self.embeddings
            clusters, centers = cop_kmeans(dataset=embeddings, k=k, ml=ml, cl=cl)

            assignment_conf = []
            distance_matrix = pairwise_distances(embeddings, centers)
            for i, u in enumerate(self.utterances):
                if u[2] == 'TRAIN':
                    assignment_conf.append(1.0)
                else:
                    scaled_dist = [1 / (1 + pow(d, 2)) for d in distance_matrix[i]]
                    assignment_conf.append(scaled_dist[clusters[i]] / sum(scaled_dist))

            if not including_train:
                clusters = [c for i, c in enumerate(clusters) if self.utterances[i][2] != 'TRAIN']
                assignment_conf = [c for i, c in enumerate(assignment_conf) if self.utterances[i][2] != 'TRAIN']
            return clusters, assignment_conf
        else:
            raise Exception('Method {} not supported'.format(method))

    def plot(self, show_train_dev_only=False, show_test_only=False, show_labels=True, show_sample_type=True,
             plot_3d=False, output_file_path=None):
        if show_train_dev_only:
            labels = [u[1] for u in self.utterances]
            sample_type = [u[2] for u in self.utterances]
            umap_plot(self.embeddings, labels, sample_type if show_sample_type else None,
                      title=self.dataset_name, show_labels=show_labels, plot_3d=plot_3d,
                      label_plotting_order=self.label_plotting_order, output_file_path=output_file_path)
        elif show_test_only:
            test_labels = [u[1] for u in self.test_utterances]

            umap_plot(self.test_embeddings, test_labels, title=self.dataset_name, show_labels=show_labels,
                      plot_3d=plot_3d,
                      label_plotting_order=self.label_plotting_order, output_file_path=output_file_path)
        else:
            embeddings = numpy.concatenate([self.embeddings, self.test_embeddings if self.use_dev else numpy.ndarray(
                (0, self.embeddings.shape[1]))])

            labels = [u[1] for u in self.utterances] + ([u[1] for u in self.test_utterances] if self.use_dev else [])
            sample_type = [u[2] for u in self.utterances] + (
                [u[2] for u in self.test_utterances] if self.use_dev else [])

            umap_plot(embeddings, labels, sample_type if show_sample_type else None,
                      title=self.dataset_name, show_labels=show_labels, plot_3d=plot_3d,
                      label_plotting_order=self.label_plotting_order, output_file_path=output_file_path)

    def fine_tune_pseudo_classification(self, k=None, use_sample_weights=False):
        pseudo_clusters, weights = self.get_pseudo_clusters(
            k=k if k is not None else len(self.cluster_label_2_index_map))
        print('Pseudo-cluster quality:',
              get_clustering_quality(self.get_true_clusters(), pseudo_clusters))
        sbert.fine_tune_pseudo_classification([u[0] for u in self.utterances], pseudo_clusters,
                                              train_sample_weights=weights if use_sample_weights else None)

    def fine_tune_utterance_similarity(self, n_train_epochs=-1, n_train_steps=-1):
        cluster_indices = [u[1] if u[2] == 'TRAIN' else None for u in self.utterances]
        sbert.fine_tune_utterance_similarity([u[0] for u in self.utterances], cluster_indices,
                                             n_train_epochs=n_train_epochs, n_train_steps=n_train_steps)

    def fine_tune_slot_recognition(self, n_train_epochs=-1, n_train_steps=-1):
        sbert.fine_tune_slot_recognition([u[0] for u in self.utterances if u[2] == 'TRAIN'],
                                         [u[3] for u in self.utterances if u[2] == 'TRAIN'],
                                         n_train_epochs=n_train_epochs, n_train_steps=n_train_steps)

    def fine_tune_joint_slot_recognition_and_utterance_similarity(self, n_train_epochs=-1, n_train_steps=-1):
        cluster_indices = [u[1] if u[2] == 'TRAIN' else None for u in self.utterances]
        sbert.fine_tune_joint_slot_recognition_and_utterance_similarity(
            [u[0] for u in self.utterances],
            [u[3] if u[2] == 'TRAIN' else None for u in self.utterances],
            cluster_indices,
            n_train_epochs=n_train_epochs, n_train_steps=n_train_steps)

    def get_test_clustering_quality(self, k=None):
        if self.use_dev:
            # Use KMeans. we haven't seen the testing utterances yet, so we use normal KMeans here.
            # The model is fully trained, and we don't want to touch the train set anymore.
            test_cluster_label_2_index_map = dict(
                (l, i) for i, l in enumerate(set([u[1] for u in self.test_utterances])))
            test_true_clusters = [test_cluster_label_2_index_map[u[1]] for u in self.test_utterances]

            if self.normalize_embeddings:
                scaler = StandardScaler()
                scaler.fit(self.test_embeddings)
                embeddings = scaler.transform(self.test_embeddings)
            else:
                embeddings = self.test_embeddings

            test_predicted_clusters = KMeans(n_clusters=k if k is not None else len(test_cluster_label_2_index_map),
                                             random_state=0).fit(embeddings).labels_

            return get_clustering_quality(test_true_clusters, test_predicted_clusters)
        else:
            # Use COP-KMeans. In this setting, we cluster the testing set, coupling with constraints from the train set.
            # So we use COP-KMeans here, which is similar to pseudo-classification.
            test_predicted_clusters = self.get_pseudo_clusters(
                k=k if k is not None else len(self.cluster_label_2_index_map), including_train=False)[0]
            return get_clustering_quality(self.get_true_clusters(including_train=False), test_predicted_clusters)

        # TODO: other clustering algorithms could be also applied here as well, e.g., C-DBScan, HAC.

    def run(self, report_folder=None, config={}, steps=['no', 'SR+US', 'PC']):
        for s in steps:
            if s not in ['no', 'SR+US', 'SR', 'US', 'PC']:
                raise Exception('Invalid step name:', s)

        sbert.load()

        stats_file = None
        if report_folder is not None:
            os.makedirs(report_folder, exist_ok=True)
            stats_file = open(os.path.join(report_folder, 'stats.txt'), 'w')

        if 'no' in steps:
            print('==================== Step: no-finetune ====================')
            folder = None
            if report_folder is not None:
                folder = os.path.join(report_folder, 'no')
                os.makedirs(folder, exist_ok=True)

            # Testing
            self.update_embeddings()
            self.update_test_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '0.pdf') if folder is not None else None)
            self.plot(show_test_only=True,
                      output_file_path=os.path.join(folder, 'test.pdf') if folder is not None else None)

            test_quality = self.get_test_clustering_quality()
            print('No-finetune test quality:', test_quality)
            if stats_file is not None:
                stats_file.write('No-finetune test quality: {}\n'.format(test_quality))

        if 'SR+US' in steps:
            print('==================== Step: finetune-slot-recognition+utterance-similarity ====================')
            folder = None
            if report_folder is not None:
                folder = os.path.join(report_folder, 'SR+US')
                os.makedirs(folder, exist_ok=True)

            self.update_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '0.pdf') if folder is not None else None)
            print('Clustering DEV(unseen) before fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))
            # Fine-tuning
            self.fine_tune_joint_slot_recognition_and_utterance_similarity()
            self.update_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '1.pdf') if folder is not None else None)

            print('Clustering DEV(unseen) after fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))

            # Testing
            self.update_test_embeddings()
            self.plot(show_test_only=True,
                      output_file_path=os.path.join(folder, 'test.pdf') if folder is not None else None)

            test_quality = self.get_test_clustering_quality()
            print('Finetune-slot-recognition+utterance-similarity test quality:', test_quality)
            if stats_file is not None:
                stats_file.write('Finetune-slot-recognition+utterance-similarity test quality: {}\n'.format(test_quality))

        if 'SR' in steps:
            print('==================== Step: finetune-slot-recognition ====================')
            folder = None
            if report_folder is not None:
                folder = os.path.join(report_folder, 'SR')
                os.makedirs(folder, exist_ok=True)

            self.update_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '0.pdf') if folder is not None else None)
            print('Clustering DEV(unseen) before fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))
            # Fine-tuning
            self.fine_tune_slot_recognition()
            self.update_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '1.pdf') if folder is not None else None)

            print('Clustering DEV(unseen) after fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))

            # Testing
            self.update_test_embeddings()
            self.plot(show_test_only=True,
                      output_file_path=os.path.join(folder, 'test.pdf') if folder is not None else None)

            test_quality = self.get_test_clustering_quality()
            print('Finetune-slot-recognition test quality:', test_quality)
            if stats_file is not None:
                stats_file.write('Finetune-slot-recognition test quality: {}\n'.format(test_quality))

        if 'US' in steps:
            print('==================== Step: finetune-utterance-similarity ====================')
            folder = None
            if report_folder is not None:
                folder = os.path.join(report_folder, 'US')
                os.makedirs(folder, exist_ok=True)

            self.update_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '0.pdf') if folder is not None else None)
            print('Clustering DEV(unseen) before fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))
            # Fine-tuning
            self.fine_tune_utterance_similarity()
            self.update_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '1.pdf') if folder is not None else None)

            print('Clustering DEV(unseen) after fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))

            # Testing
            self.update_test_embeddings()
            self.plot(show_test_only=True,
                      output_file_path=os.path.join(folder, 'test.pdf') if folder is not None else None)

            test_quality = self.get_test_clustering_quality()
            print('Finetune-utterance-similarity test quality:', test_quality)
            if stats_file is not None:
                stats_file.write('Finetune-utterance-similarity test quality: {}\n'.format(test_quality))

        if 'PC' in steps:
            print('==================== Step: finetune-pseudo-classification ====================')
            folder = None
            if report_folder is not None:
                folder = os.path.join(report_folder, 'PC')
                os.makedirs(folder, exist_ok=True)

            self.update_embeddings()
            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(folder, '0.pdf') if folder is not None else None)
            print('Clustering DEV(unseen) before fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))
            # Fine-tuning
            for it in range(5):
                print('Iter: #{}'.format(it + 1))
                self.fine_tune_pseudo_classification(
                    use_sample_weights=(('classification_sample_weights', True) in config.items()))
                self.update_embeddings()
                self.plot(show_train_dev_only=True,
                          output_file_path=os.path.join(folder,
                                                        '{}.pdf'.format(it + 1)) if folder is not None else None)

            print('Clustering DEV(unseen) after fine-tuning:',
                  get_clustering_quality(self.get_true_clusters(including_train=False),
                                         self.get_pseudo_clusters(k=len(self.cluster_label_2_index_map),
                                                                  including_train=False)[0]))

            # Testing
            self.update_test_embeddings()
            self.plot(show_test_only=True,
                      output_file_path=os.path.join(folder, 'test.pdf') if folder is not None else None)

            test_quality = self.get_test_clustering_quality()
            print('Finetune-pseudo-classification test quality:', test_quality)
            if stats_file is not None:
                stats_file.write('Finetune-pseudo-classification test quality: {}\n'.format(test_quality))

        if stats_file is not None:
            stats_file.close()
