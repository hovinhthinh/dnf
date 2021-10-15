import tempfile

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
from transformers import set_seed

Axes3D = Axes3D

import os
from math import pow
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances, euclidean_distances

import sbert
from cluster import cop_kmeans, get_clustering_quality
from data.snips import print_train_dev_test_stats


def umap_plot(embeddings, labels, sample_type=None, title=None, show_labels=False, plot_3d=False,
              label_plotting_order=None, output_file_path=None):
    if show_labels:
        plt.rcParams["figure.figsize"] = (10, 4)
    embeddings = umap.UMAP(n_components=3 if plot_3d else 2, random_state=42).fit_transform(embeddings)
    ax = plt.figure().add_subplot(projection='3d' if plot_3d else None)

    u_labels = dict.fromkeys(labels)
    if label_plotting_order is None:
        label_plotting_order = [(l, None) for l in u_labels]

    for l, lc in label_plotting_order:
        if l not in u_labels:
            continue
        idx = [i for i, _ in enumerate(labels) if _ == l]
        l = l.replace('_TRAIN', '_TRAIN_L').replace('_DEV', '_TRAIN_U')  # rename cluster names
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
                 dataset_name=None, squashing_train_dev=False, use_unseen_in_training=True,
                 dev_test_clustering_method='k-means'):
        self.dataset_name = dataset_name
        self.use_unseen_in_training = use_unseen_in_training
        self.dev_test_clustering_method = dev_test_clustering_method
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

        # For squashing train/dev.
        self.squashing_train_dev = squashing_train_dev
        self.dev_indices = [i for i, u in enumerate(self.utterances) if u[2] != 'TRAIN']

        self.cluster_label_2_index_map = dict(
            (n, i) for i, n in enumerate(dict.fromkeys([u[1] for u in self.utterances])))

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

        # Squashing TRAIN and DEV. This will disable negative sampling and also take into account the labels/slots of
        # DEV utterances during training.
        if squashing_train_dev:
            if not self.use_dev:
                raise Exception('squashing_train_dev is only possible when use_dev is True')
            self.utterances = [list(u) for u in self.utterances]
            for u in self.utterances:
                u[2] = 'TRAIN'

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

    def get_true_clusters(self, including_train=True, including_dev=True):
        return [self.cluster_label_2_index_map[u[1]] for u in self.utterances if
                (including_train and u[2] == 'TRAIN') or (including_dev and u[2] != 'TRAIN')]

    # Returns pseudo clusters and assignment confidences
    def get_pseudo_clusters(self, k=None):
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

        clusters, centers = cop_kmeans(dataset=self.embeddings,
                                       k=k if k is not None else len(self.cluster_label_2_index_map), ml=ml, cl=cl)
        assignment_conf = []
        distance_matrix = pairwise_distances(self.embeddings, centers)
        for i, u in enumerate(self.utterances):
            if u[2] == 'TRAIN':
                assignment_conf.append(1.0)
            else:
                scaled_dist = [1 / (1 + pow(d, 2)) for d in distance_matrix[i]]
                assignment_conf.append(scaled_dist[clusters[i]] / sum(scaled_dist))
        return clusters, assignment_conf

    # pseudo_cluster_0: previous clusters, pseudo_cluster_1: current clusters
    def get_aligned_pseudo_clusters(self, embeddings, pseudo_clusters_0, pseudo_clusters_1):
        assert max(pseudo_clusters_0) == max(pseudo_clusters_1)
        assert len(embeddings) == len(pseudo_clusters_0) == len(pseudo_clusters_1)

        k = max(pseudo_clusters_0) + 1
        centers_0 = numpy.zeros((k, len(embeddings[0])))
        counts_0 = numpy.zeros(k)
        centers_1 = numpy.zeros((k, len(embeddings[0])))
        counts_1 = numpy.zeros(k)
        for i in range(0, len(pseudo_clusters_0)):
            counts_0[pseudo_clusters_0[i]] += 1
            centers_0[pseudo_clusters_0[i]] += embeddings[i]
            counts_1[pseudo_clusters_1[i]] += 1
            centers_1[pseudo_clusters_1[i]] += embeddings[i]

        for i in range(k):
            centers_0[i] /= counts_0[i]
            centers_1[i] /= counts_1[i]

        distance_matrix = euclidean_distances(centers_0, centers_1)

        _, forward_mapping = linear_sum_assignment(distance_matrix)
        backward_mapping = numpy.zeros((k,), dtype=int)
        for i in range(k):
            backward_mapping[forward_mapping[i]] = i
        return [backward_mapping[i] for i in pseudo_clusters_1]

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

    def get_validation_score(self):
        self.update_embeddings()
        return self.get_dev_clustering_quality()['NMI']

    # map to 0,1,2...
    def remap_clusters(self, clusters):
        label_set = dict.fromkeys(clusters)
        label_map = {l: i for i, l in enumerate(label_set)}
        return [label_map[l] for l in clusters]

    def fine_tune_pseudo_classification(self, use_sample_weights=True, iterations=None, align_clusters=True,
                                        early_stopping_patience=0, min_iterations=None, max_iterations=None):
        classifier, optim, previous_clusters = None, None, None
        if iterations is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                best_iter = None
                best_eval = None
                it = 0
                while True:
                    it += 1
                    print('==== Iteration: {}'.format(it))
                    # self.update_embeddings() # No need to update, already called in self.get_validation_score()
                    utterances = self.utterances
                    embeddings = self.embeddings
                    pseudo_clusters, weights = self.get_pseudo_clusters()

                    if not self.use_unseen_in_training:
                        utterances = [u for u in utterances if u[2] == 'TRAIN']
                        embeddings = [embeddings[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                        pseudo_clusters = [pseudo_clusters[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                        weights = [weights[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']

                    pseudo_clusters = self.remap_clusters(pseudo_clusters)

                    if align_clusters and previous_clusters is not None:
                        pseudo_clusters = self.get_aligned_pseudo_clusters(embeddings, previous_clusters,
                                                                           pseudo_clusters)
                    previous_clusters = pseudo_clusters

                    print('Pseudo-cluster quality:',
                          get_clustering_quality(self.get_true_clusters(including_dev=self.use_unseen_in_training),
                                                 pseudo_clusters))
                    classifier, optim = sbert.fine_tune_pseudo_classification([u[0] for u in utterances],
                                                                              pseudo_clusters,
                                                                              train_sample_weights=weights if use_sample_weights else None,
                                                                              previous_classifier=classifier if align_clusters else None,
                                                                              previous_optim=optim if align_clusters else None)
                    eval = self.get_validation_score()
                    print('Validation score: {:.3f}'.format(eval), end='')
                    if best_eval is None or eval > best_eval:
                        best_eval = eval
                        best_iter = it
                        print(' -> Save model', end='')
                        sbert.save(temp_dir)

                    if (it > best_iter + early_stopping_patience and (
                            min_iterations is None or it >= min_iterations)) \
                            or (max_iterations is not None and it == max_iterations):
                        print(' -> Stop')
                        sbert.load(temp_dir)
                        break
                    else:
                        print()
        else:
            for it in range(iterations):
                print('Iter: {}'.format(it + 1))
                # self.update_embeddings() # No need to update
                utterances = self.utterances
                embeddings = self.embeddings
                pseudo_clusters, weights = self.get_pseudo_clusters()

                if not self.use_unseen_in_training:
                    utterances = [u for u in utterances if u[2] == 'TRAIN']
                    embeddings = [embeddings[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                    pseudo_clusters = [pseudo_clusters[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                    weights = [weights[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']

                pseudo_clusters = self.remap_clusters(pseudo_clusters)

                if align_clusters and previous_clusters is not None:
                    pseudo_clusters = self.get_aligned_pseudo_clusters(embeddings, previous_clusters, pseudo_clusters)
                previous_clusters = pseudo_clusters

                print('Pseudo-cluster quality:',
                      get_clustering_quality(self.get_true_clusters(including_dev=self.use_unseen_in_training),
                                             pseudo_clusters))
                classifier, optim = sbert.fine_tune_pseudo_classification([u[0] for u in utterances], pseudo_clusters,
                                                                          train_sample_weights=weights if use_sample_weights else None,
                                                                          previous_classifier=classifier if align_clusters else None,
                                                                          previous_optim=optim if align_clusters else None)
                print('Validation score: {:.3f}'.format(self.get_validation_score()))

    def fine_tune_joint_pseudo_classification_and_intent_classification(
            self, use_pseudo_sample_weights=True, align_clusters=True, intent_classifier_weight=0.1,
            iterations=None, early_stopping_patience=0, min_iterations=None, max_iterations=None):
        classifier, optim, previous_clusters = None, None, None
        if iterations is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                best_iter = None
                best_eval = None
                it = 0
                while True:
                    it += 1
                    print('==== Iteration: {}'.format(it))
                    # self.update_embeddings() # No need to update, already called in self.get_validation_score()
                    utterances = self.utterances
                    embeddings = self.embeddings
                    pseudo_clusters, weights = self.get_pseudo_clusters()

                    if not self.use_unseen_in_training:
                        utterances = [u for u in utterances if u[2] == 'TRAIN']
                        embeddings = [embeddings[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                        pseudo_clusters = [pseudo_clusters[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                        weights = [weights[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']

                    pseudo_clusters = self.remap_clusters(pseudo_clusters)

                    if align_clusters and previous_clusters is not None:
                        pseudo_clusters = self.get_aligned_pseudo_clusters(embeddings, previous_clusters,
                                                                           pseudo_clusters)
                    previous_clusters = pseudo_clusters

                    print('Pseudo-cluster quality:',
                          get_clustering_quality(self.get_true_clusters(including_dev=self.use_unseen_in_training),
                                                 pseudo_clusters))
                    classifier, optim = sbert.fine_tune_joint_pseudo_classification_and_intent_classification(
                        [u[0] for u in utterances], pseudo_clusters,
                        self.remap_clusters([u[4] for u in utterances]),
                        train_sample_weights=weights if use_pseudo_sample_weights else None,
                        intent_classifier_weight=intent_classifier_weight,
                        previous_classifier=classifier if align_clusters else None,
                        previous_optim=optim if align_clusters else None)
                    eval = self.get_validation_score()
                    print('Validation score: {:.3f}'.format(eval), end='')
                    if best_eval is None or eval > best_eval:
                        best_eval = eval
                        best_iter = it
                        print(' -> Save model', end='')
                        sbert.save(temp_dir)

                    if (it > best_iter + early_stopping_patience and (
                            min_iterations is None or it >= min_iterations)) \
                            or (max_iterations is not None and it == max_iterations):
                        print(' -> Stop')
                        sbert.load(temp_dir)
                        break
                    else:
                        print()
        else:
            for it in range(iterations):
                print('Iter: {}'.format(it + 1))
                # self.update_embeddings() # No need to update
                utterances = self.utterances
                embeddings = self.embeddings
                pseudo_clusters, weights = self.get_pseudo_clusters()

                if not self.use_unseen_in_training:
                    utterances = [u for u in utterances if u[2] == 'TRAIN']
                    embeddings = [embeddings[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                    pseudo_clusters = [pseudo_clusters[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']
                    weights = [weights[i] for i, u in enumerate(self.utterances) if u[2] == 'TRAIN']

                pseudo_clusters = self.remap_clusters(pseudo_clusters)

                if align_clusters and previous_clusters is not None:
                    pseudo_clusters = self.get_aligned_pseudo_clusters(embeddings, previous_clusters, pseudo_clusters)
                previous_clusters = pseudo_clusters

                print('Pseudo-cluster quality:',
                      get_clustering_quality(self.get_true_clusters(including_dev=self.use_unseen_in_training),
                                             pseudo_clusters))
                classifier, optim = sbert.fine_tune_joint_pseudo_classification_and_intent_classification(
                    [u[0] for u in utterances], pseudo_clusters,
                    self.remap_clusters([u[4] for u in utterances]),
                    intent_classifier_weight=intent_classifier_weight,
                    train_sample_weights=weights if use_pseudo_sample_weights else None,
                    previous_classifier=classifier if align_clusters else None,
                    previous_optim=optim if align_clusters else None)
                print('Validation score: {:.3f}'.format(self.get_validation_score()))

    def fine_tune_utterance_similarity(self, n_train_epochs=None):
        if self.use_unseen_in_training:
            sbert.fine_tune_utterance_similarity([u[0] for u in self.utterances],
                                                 [u[1] if u[2] == 'TRAIN' else None for u in self.utterances],
                                                 n_train_epochs=n_train_epochs,
                                                 eval_callback=self.get_validation_score,
                                                 early_stopping=True if n_train_epochs is None else False)
        else:
            sbert.fine_tune_utterance_similarity([u[0] for u in self.utterances if u[2] == 'TRAIN'],
                                                 [u[1] for u in self.utterances if u[2] == 'TRAIN'],
                                                 n_train_epochs=n_train_epochs,
                                                 eval_callback=self.get_validation_score,
                                                 early_stopping=True if n_train_epochs is None else False)

    @DeprecationWarning
    def fine_tune_slot_tagging(self):
        sbert.fine_tune_slot_tagging([u[0] for u in self.utterances],
                                     [u[3] for u in self.utterances],
                                     eval_callback=self.get_validation_score, early_stopping=True)

    def fine_tune_slot_multiclass_classification(self, n_train_epochs=None):
        sbert.fine_tune_slot_multiclass_classification([u[0] for u in self.utterances],
                                                       [u[3] for u in self.utterances],
                                                       n_train_epochs=n_train_epochs,
                                                       eval_callback=self.get_validation_score,
                                                       early_stopping=True if n_train_epochs is None else False)

    def fine_tune_joint_slot_multiclass_classification_and_utterance_similarity(self, n_train_epochs=None,
                                                                                early_stopping_patience=0):
        if self.use_unseen_in_training:
            utterances, slots, clusters = [u[0] for u in self.utterances], \
                                          [u[3] for u in self.utterances], \
                                          [u[1] if u[2] == 'TRAIN' else None for u in self.utterances]
        else:
            utterances, slots, clusters = [u[0] for u in self.utterances if u[2] == 'TRAIN'], \
                                          [u[3] for u in self.utterances if u[2] == 'TRAIN'], \
                                          [u[1] for u in self.utterances if u[2] == 'TRAIN']
        sbert.fine_tune_joint_slot_multiclass_classification_and_utterance_similarity(
            utterances, slots, clusters,
            us_loss_weight=0.5, smc_loss_weight=0.5,
            n_train_epochs=n_train_epochs,
            eval_callback=self.get_validation_score,
            early_stopping=True if n_train_epochs is None else False,
            early_stopping_patience=early_stopping_patience)

    def fine_tune_joint_slot_multiclass_classification_and_utterance_similarity_and_intent_classification(
            self, n_train_epochs=None, early_stopping_patience=0):
        if self.use_unseen_in_training:
            utterances, slots, clusters, intents = [u[0] for u in self.utterances], \
                                                   [u[3] for u in self.utterances], \
                                                   [u[1] if u[2] == 'TRAIN' else None for u in self.utterances], \
                                                   [u[4] for u in self.utterances]
        else:
            utterances, slots, clusters, intents = [u[0] for u in self.utterances if u[2] == 'TRAIN'], \
                                                   [u[3] for u in self.utterances if u[2] == 'TRAIN'], \
                                                   [u[1] for u in self.utterances if u[2] == 'TRAIN'], \
                                                   [u[4] for u in self.utterances if u[2] == 'TRAIN']
        sbert.fine_tune_joint_slot_multiclass_classification_and_utterance_similarity_and_intent_classification(
            utterances, slots, clusters, intents,
            us_loss_weight=0.4, smc_loss_weight=0.4, ic_loss_weight=0.2,
            n_train_epochs=n_train_epochs,
            eval_callback=self.get_validation_score,
            early_stopping=True if n_train_epochs is None else False,
            early_stopping_patience=early_stopping_patience)

    def get_dev_clustering_quality(self):
        if self.dev_test_clustering_method == 'k-means':
            clusterer = KMeans(n_clusters=len(self.cluster_label_2_index_map))
        elif self.dev_test_clustering_method == 'hac-complete':
            clusterer = AgglomerativeClustering(n_clusters=len(self.cluster_label_2_index_map), linkage='complete')
        else:
            raise Exception('Invalid clustering method')

        if self.squashing_train_dev:
            dev_predicted_clusters = clusterer.fit([self.embeddings[i] for i in self.dev_indices]).labels_
            dev_true_clusters = [self.cluster_label_2_index_map[self.utterances[i][1]] for i in self.dev_indices]
            return get_clustering_quality(dev_true_clusters, dev_predicted_clusters)
        else:
            dev_embeddings = [self.embeddings[i] for i, u in enumerate(self.utterances) if u[2] != 'TRAIN']
            if len(dev_embeddings) == 0:
                raise Exception('DEV set is unavailable')
            dev_predicted_clusters = clusterer.fit(dev_embeddings).labels_
            return get_clustering_quality(self.get_true_clusters(including_train=False), dev_predicted_clusters)

    def get_test_clustering_quality(self, k=None, predicted_clusters_log_file=None, true_clusters_log_file=None,
                                    contingency_matrix_log_file=None):
        true_clusters = dict.fromkeys([u[1] for u in self.test_utterances if u[1].endswith('_TRAIN')])
        true_clusters.update(dict.fromkeys([u[1] for u in self.test_utterances if u[1].endswith('_DEV')]))
        true_clusters.update(dict.fromkeys([u[1] for u in self.test_utterances if u[1].endswith('_TEST')]))

        true_clusters = [l for l in true_clusters]
        true_cluster_2_index_map = dict((l, i) for i, l in enumerate(true_clusters))
        test_true_clusters = [true_cluster_2_index_map[u[1]] for u in self.test_utterances]

        k = k if k is not None else len(true_cluster_2_index_map)

        if self.dev_test_clustering_method == 'k-means':
            test_predicted_clusters = KMeans(n_clusters=k).fit(self.test_embeddings).labels_
        elif self.dev_test_clustering_method == 'hac-complete':
            test_predicted_clusters = AgglomerativeClustering(n_clusters=k, linkage='complete').fit(
                self.test_embeddings).labels_
        else:
            raise Exception('Invalid clustering method')

        if predicted_clusters_log_file is not None:
            with open(predicted_clusters_log_file, 'w') as f:
                for c in range(k):
                    indices = [i for i, l in enumerate(test_predicted_clusters) if l == c]
                    f.write('==== Cluster: {} | {}/{} ({:.1f}%)\n'
                            .format(c, len(indices), len(self.test_utterances),
                                    len(indices) / len(self.test_utterances) * 100))

                    true_label_2_count = {}
                    for idx in indices:
                        tl = test_true_clusters[idx]
                        true_label_2_count[tl] = true_label_2_count.get(tl, 0) + 1

                    for tl, cnt in sorted(true_label_2_count.items(), key=lambda o: o[1], reverse=True):
                        f.write('Feature: {} | {} ({:.1f}%)\n'.format(true_clusters[tl], cnt, cnt / len(indices) * 100))
                        f.write('    {}\n'.format(
                            [self.test_utterances[idx][0] for idx in indices if test_true_clusters[idx] == tl]))
                    f.write('\n')

        if true_clusters_log_file is not None:
            with open(true_clusters_log_file, 'w') as f:
                for c in range(len(true_clusters)):
                    indices = [i for i, l in enumerate(test_true_clusters) if l == c]
                    f.write('==== Feature: {} | {}/{} ({:.1f}%)\n'
                            .format(true_clusters[c], len(indices), len(self.test_utterances),
                                    len(indices) / len(self.test_utterances) * 100))

                    predicted_label_2_count = {}
                    for idx in indices:
                        pl = test_predicted_clusters[idx]
                        predicted_label_2_count[pl] = predicted_label_2_count.get(pl, 0) + 1

                    for pl, cnt in sorted(predicted_label_2_count.items(), key=lambda o: o[1], reverse=True):
                        f.write('Cluster: {} | {} ({:.1f}%)\n'.format(pl, cnt, cnt / len(indices) * 100))
                        f.write('    {}\n'.format(
                            [self.test_utterances[idx][0] for idx in indices if test_predicted_clusters[idx] == pl]))
                    f.write('\n')

        if contingency_matrix_log_file is not None:
            with open(contingency_matrix_log_file, 'w') as f:
                matrix = numpy.zeros((len(true_clusters), k), dtype=numpy.int32)
                for tc in range(len(true_clusters)):
                    indices = [i for i, l in enumerate(test_true_clusters) if l == tc]
                    for idx in indices:
                        matrix[tc][test_predicted_clusters[idx]] += 1

                sum_by_true_clusters = numpy.sum(matrix, axis=1)
                sum_by_predicted_clusters = numpy.sum(matrix, axis=0)

                f.write('Feature/Cluster')
                for i in range(k):
                    f.write(
                        '\tC_{} ({:.1f}%)'.format(i, sum_by_predicted_clusters[i] / len(self.test_utterances) * 100))
                f.write('\tPurity\n')

                for i, tc in enumerate(true_clusters):
                    f.write('{} ({:.1f}%)'.format(tc, sum_by_true_clusters[i] / len(self.test_utterances) * 100))
                    for j in range(k):
                        f.write('\t{}'.format(matrix[i][j]))
                    f.write('\t{:.1f}%\n'.format(numpy.max(matrix[i]) / sum_by_true_clusters[i] * 100))

                f.write('Purity')
                for i in range(k):
                    f.write('\t{:.1f}%'.format(numpy.max(matrix[:, i]) / sum_by_predicted_clusters[i] * 100))
                f.write('\t\n')

        return {
            'all': get_clustering_quality(test_true_clusters, test_predicted_clusters),
            'train_dev': get_clustering_quality(
                [test_true_clusters[i] for i, u in enumerate(self.test_utterances) if
                 u[1].endswith('_TRAIN') or u[1].endswith('_DEV')],
                [test_predicted_clusters[i] for i, u in enumerate(self.test_utterances) if
                 u[1].endswith('_TRAIN') or u[1].endswith('_DEV')]),
            'test': get_clustering_quality(
                [test_true_clusters[i] for i, u in enumerate(self.test_utterances) if u[1].endswith('_TEST')],
                [test_predicted_clusters[i] for i, u in enumerate(self.test_utterances) if u[1].endswith('_TEST')])
        }
        # TODO: other clustering algorithms could be also applied here as well, e.g., DBScan, HAC.

    def run(self, report_folder=None, steps=['SMC+US', 'PC'], save_model=True, plot_3d=False,
            config={
                'SMC_n_train_epochs': None,
                'US_n_train_epochs': None,
                'SMC+US_n_train_epochs': None,
                'IC+SMC+US_n_train_epochs': None,
                'PC_sample_weights': True,
                'PC_iterations': None,
                'PC_max_iterations': 10,
            }):
        set_seed(12993)
        for s in steps:
            if s not in ['SMC+US', 'IC+SMC+US', 'PC+IC', 'ST', 'SMC', 'US', 'PC']:
                raise Exception('Invalid step name:', s)

        sbert.load()

        stats_file = None
        if report_folder is not None:
            os.makedirs(report_folder, exist_ok=True)
            stats_file = open(os.path.join(report_folder, 'stats.txt'), 'w')

        # No-fine-tuning
        print('==================== Step: no-finetune ====================')
        folder_3d = None
        if plot_3d and report_folder is not None:
            folder_3d = os.path.join(report_folder, '3d')
            os.makedirs(folder_3d, exist_ok=True)

        # Testing for no-fine-tuning
        self.update_embeddings()
        self.update_test_embeddings()
        self.plot(show_train_dev_only=True,
                  output_file_path=os.path.join(report_folder, '0.pdf') if report_folder is not None else None)
        self.plot(show_test_only=True,
                  output_file_path=os.path.join(report_folder, '0_test.pdf') if report_folder is not None else None)

        if folder_3d is not None:
            self.plot(show_train_dev_only=True, plot_3d=True, output_file_path=os.path.join(folder_3d, '0.pdf'))
            self.plot(show_test_only=True, plot_3d=True, output_file_path=os.path.join(folder_3d, '0_test.pdf'))

        print('Clustering DEV no-fine-tuning:', self.get_dev_clustering_quality())

        test_quality = self.get_test_clustering_quality(
            predicted_clusters_log_file=
            os.path.join(report_folder, '0_test.predicted_clusters.log') if report_folder is not None else None,
            true_clusters_log_file=
            os.path.join(report_folder, '0_test.true_clusters.log') if report_folder is not None else None,
            contingency_matrix_log_file=
            os.path.join(report_folder, '0_test.contingency_matrix.tsv') if report_folder is not None else None)
        print('No-fine-tune test quality:', test_quality)
        if stats_file is not None:
            stats_file.write('No-fine-tune test quality: {}\n'.format(test_quality))

        # Fine-tuning each step
        for i, step in enumerate(steps):
            print('==================== Step: finetune-{} ===================='.format(step))

            # Fine-tuning
            if step == 'IC+SMC+US':
                self.fine_tune_joint_slot_multiclass_classification_and_utterance_similarity_and_intent_classification(
                    early_stopping_patience=3,
                    n_train_epochs=config.get('IC+SMC+US_n_train_epochs', None))
            elif step == 'SMC+US':
                self.fine_tune_joint_slot_multiclass_classification_and_utterance_similarity(
                    early_stopping_patience=3,
                    n_train_epochs=config.get('SMC+US_n_train_epochs', None))
            elif step == 'ST':
                self.fine_tune_slot_tagging()
            elif step == 'SMC':
                self.fine_tune_slot_multiclass_classification(n_train_epochs=config.get('SMC_n_train_epochs', None))
            elif step == 'US':
                self.fine_tune_utterance_similarity(n_train_epochs=config.get('US_n_train_epochs', None))
            elif step == 'PC':
                self.fine_tune_pseudo_classification(
                    use_sample_weights=config.get('PC_sample_weights', True),
                    iterations=config.get('PC_iterations', None),
                    early_stopping_patience=3,
                    min_iterations=1, max_iterations=config.get('PC_max_iterations', 10),
                )
            elif step == 'PC+IC':
                self.fine_tune_joint_pseudo_classification_and_intent_classification(
                    use_pseudo_sample_weights=config.get('PC_sample_weights', True),
                    iterations=config.get('PC_iterations', None),
                    early_stopping_patience=3,
                    min_iterations=1, max_iterations=config.get('PC_max_iterations', 10),
                )
            else:
                raise Exception('Invalid step name:', step)

            self.update_embeddings()

            self.plot(show_train_dev_only=True,
                      output_file_path=os.path.join(report_folder, '{}_{}.pdf'.format(i + 1, step))
                      if report_folder is not None else None)
            if folder_3d is not None:
                self.plot(show_train_dev_only=True, plot_3d=True,
                          output_file_path=os.path.join(folder_3d, '{}_{}.pdf'.format(i + 1, step)))

            print('Clustering DEV after fine-tuning {}: {}'.format(step, self.get_dev_clustering_quality()))

            # Testing
            self.update_test_embeddings()
            self.plot(show_test_only=True,
                      output_file_path=os.path.join(report_folder, '{}_{}_test.pdf'.format(i + 1, step))
                      if report_folder is not None else None)
            if folder_3d is not None:
                self.plot(show_test_only=True, plot_3d=True,
                          output_file_path=os.path.join(folder_3d, '{}_{}_test.pdf'.format(i + 1, step)))

            test_quality = self.get_test_clustering_quality(
                predicted_clusters_log_file=
                os.path.join(report_folder, '{}_{}_test.predicted_clusters.log'.format(i + 1, step))
                if report_folder is not None else None,
                true_clusters_log_file=
                os.path.join(report_folder, '{}_{}_test.true_clusters.log'.format(i + 1, step))
                if report_folder is not None else None,
                contingency_matrix_log_file=
                os.path.join(report_folder, '{}_{}_test.contingency_matrix.tsv'.format(i + 1, step))
                if report_folder is not None else None)
            print('Finetune-{} test quality: {}'.format(step, test_quality))
            if stats_file is not None:
                stats_file.write('Finetune-{} TEST quality: {}\n'.format(step, test_quality))

        if stats_file is not None:
            stats_file.close()

        if save_model and report_folder is not None:
            sbert.save(os.path.join(report_folder, 'trained_model'))


def run_all_intents(pipeline_steps, intra_intent_data, inter_intent_data,
                    report_folder=None, plot_3d=False,
                    config={
                        'use_unseen_in_training': True,
                        'squashing_train_dev': False,
                        'dev_test_clustering_method': 'k-means'
                    }):
    # Processing intra-intents
    if intra_intent_data is not None:
        for intent_name, intent_data in intra_intent_data:
            print('======================================== Intra-intent:', intent_name,
                  '========================================')

            if len([u for u in intent_data if u[1].endswith('_TRAIN')]) == 0 or len(
                    [u for u in intent_data if u[1].endswith('_TEST')]) == 0:
                print('Ignore this intent for intra-intent setting')
                continue

            print_train_dev_test_stats(intent_data)
            p = Pipeline(intent_data, dataset_name=intent_name,
                         use_unseen_in_training=config.get('use_unseen_in_training', True),
                         squashing_train_dev=config.get('squashing_train_dev', False),
                         dev_test_clustering_method=config.get('dev_test_clustering_method', 'k-means'))
            intent_report_folder = os.path.join(report_folder, intent_name) if report_folder is not None else None
            p.run(report_folder=intent_report_folder, steps=pipeline_steps, config=config, plot_3d=plot_3d)

    # Processing inter-intents
    if inter_intent_data is not None:
        print('======================================== Inter-intent ======================================')
        print_train_dev_test_stats(inter_intent_data)
        p = Pipeline(inter_intent_data, dataset_name='inter_intent',
                     use_unseen_in_training=config.get('use_unseen_in_training', True),
                     squashing_train_dev=config.get('squashing_train_dev', False),
                     dev_test_clustering_method=config.get('dev_test_clustering_method', 'k-means'))
        intent_report_folder = os.path.join(report_folder, 'inter_intent') if report_folder is not None else None
        p.run(report_folder=intent_report_folder, steps=pipeline_steps, config=config, plot_3d=plot_3d)

        if intra_intent_data is not None:
            # Apply back to intra-intent
            print('======== Apply inter-intent model back to intra-intent ========')
            folder = None
            folder_3d = None
            if report_folder is not None:
                folder = os.path.join(report_folder, 'inter_intent', 'apply_to_intra_intent')
                os.makedirs(folder, exist_ok=True)
                if plot_3d:
                    folder_3d = os.path.join(folder, '3d')
                    os.makedirs(folder_3d, exist_ok=True)

            stats_file = None
            if folder is not None:
                stats_file = open(os.path.join(folder, 'stats.txt'), 'w')
                stats_file.write('======== Apply inter-intent model back to intra-intent ========\n')

            for intent_name, intent_data in intra_intent_data:
                p = Pipeline(intent_data, dataset_name=intent_name,
                             dev_test_clustering_method=config.get('dev_test_clustering_method', 'k-means'))
                p.update_test_embeddings()
                p.plot(show_test_only=True,
                       output_file_path=os.path.join(folder, intent_name + '.pdf') if folder is not None else None)
                if folder_3d is not None:
                    p.plot(show_test_only=True, plot_3d=True,
                           output_file_path=os.path.join(folder_3d, intent_name + '.pdf'))
                test_quality = p.get_test_clustering_quality(
                    predicted_clusters_log_file=os.path.join(folder, intent_name + '.predicted_clusters.log')
                    if folder is not None else None,
                    true_clusters_log_file=os.path.join(folder, intent_name + '.true_clusters.log')
                    if folder is not None else None,
                    contingency_matrix_log_file=os.path.join(folder, intent_name + '.contingency_matrix.tsv')
                    if folder is not None else None)
                print('Clustering TEST quality [{}]: {}'.format(intent_name, test_quality))
                if stats_file is not None:
                    stats_file.write('Clustering TEST quality [{}]: {}\n'.format(intent_name, test_quality))

            if stats_file is not None:
                stats_file.close()
