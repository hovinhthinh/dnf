import random
from typing import List

import numpy
from scipy.optimize import linear_sum_assignment
from sklearn.cluster._kmeans import _tolerance
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, \
    fowlkes_mallows_score


# Disjoint set
def _get_root(dad: List[int], u):
    if dad[u] < 0:
        return u
    dad[u] = _get_root(dad, dad[u])
    return dad[u]


def _union(dad: List[int], root_u, root_v):
    if dad[root_u] >= 0 or dad[root_v] >= 0:
        raise Exception("Either u or v is not a tree root")
    if root_u == root_v:
        return
    if dad[root_u] < dad[root_v]:
        dad[root_u] += dad[root_v]
        dad[root_v] = root_u
    else:
        dad[root_v] += dad[root_u]
        dad[root_u] = root_v


def cop_kmeans(dataset, k, ml=[], cl=[], initialization='kmpp', max_iter=100, tol=1e-4):
    n = len(dataset)
    dad = [-1] * n
    for u, v in ml:
        _union(dad, _get_root(dad, u), _get_root(dad, v))

    root_2_idx = {}
    for i in range(n):
        r = _get_root(dad, i)
        if r not in root_2_idx:
            root_2_idx[r] = []
        root_2_idx[r].append(i)

    cl_root = {}
    for u, v in cl:
        ru = _get_root(dad, u)
        rv = _get_root(dad, v)
        if ru == rv:
            raise Exception('Inconsistent constraints between %d and %d' % (u, v))
        if ru not in cl_root:
            cl_root[ru] = set()
        cl_root[ru].add(rv)
        if rv not in cl_root:
            cl_root[rv] = set()
        cl_root[rv].add(ru)

    ml_info = _get_ml_info(root_2_idx, dataset)
    tol = _tolerance(dataset, tol)

    centers = _initialize_centers(dataset, k, initialization)
    cls = [-1] * n

    for _ in range(max_iter):
        print('\rCOP-KMeans iteration:', _ + 1, end='')
        clusters_ = [-1] * n
        for i, d in enumerate(dataset):
            if clusters_[i] == -1:
                indices, _ = _closest_clusters(centers, d)
                counter = 0
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    r = _get_root(dad, i)
                    index = indices[counter]

                    violate = False
                    if r in cl_root:
                        for r2 in cl_root[r]:
                            if clusters_[r2] == index:
                                violate = True
                                break

                    if not violate:
                        found_cluster = True
                        for j in root_2_idx[r]:
                            clusters_[j] = index
                    counter += 1

                if not found_cluster:
                    print()
                    return None, None

        clusters_, centers_ = _compute_centers(clusters_, dataset, k, ml_info)
        shift = numpy.sum((centers - centers_) ** 2)
        if shift <= tol or _relabel(clusters_) == _relabel(cls):
            break

        centers = centers_
        cls = clusters_

    print()
    return clusters_, centers_


def _relabel(cls):
    cluster_map = {}
    cluster_count = 0
    for c in cls:
        if c not in cluster_map:
            cluster_map[c] = cluster_count
            cluster_count += 1
    return [cluster_map[c] for c in cls]


def _l2_distance(point1, point2):
    return numpy.sum((point1 - point2) ** 2)


def _closest_clusters(centers, datapoint):
    distances = [_l2_distance(center, datapoint) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances


def _initialize_centers(dataset, k, method):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]

    elif method == 'kmpp':
        chances_raw = numpy.asarray([1.0] * len(dataset))
        centers = []
        for it in range(k):
            chances = chances_raw / numpy.sum(chances_raw)
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                acc += chance
                if acc >= r:
                    break
            centers.append(dataset[index])

            for index, point in enumerate(dataset):
                d = _l2_distance(point, centers[-1])
                if it == 0 or chances_raw[index] > d:
                    chances_raw[index] = d

        return centers


def _compute_centers(clusters, dataset, k, ml_info):
    cluster_ids = dict.fromkeys(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]

    centers = numpy.zeros((k, len(dataset[0])))
    counts = numpy.zeros((k_new))
    for j, c in enumerate(clusters):
        centers[c] += dataset[j]
        counts[c] += 1
    for j in range(k_new):
        centers[j] /= counts[j]

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [sum(_l2_distance(centers[clusters[i]], dataset[i]) for i in group) for group in ml_groups]
        group_ids = sorted(range(len(ml_groups)), key=lambda x: current_scores[x] - ml_scores[x], reverse=True)

        for j in range(k - k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    return clusters, centers


def _get_ml_info(root_2_idx, dataset):
    groups = list(root_2_idx.values())
    centroids = numpy.zeros((len(groups), len(dataset[0])))

    for j, group in enumerate(groups):
        for i in group:
            centroids[j] += dataset[i]
        centroids[j] /= float(len(group))

    scores = [sum(_l2_distance(centroids[j], dataset[i]) for i in group) for j, group in enumerate(groups)]

    return groups, scores, centroids


def get_clustering_quality(labels_true, labels_pred):
    quality = {
        'NMI': round(normalized_mutual_info_score(labels_true, labels_pred), 3),
        'ARI': round(adjusted_rand_score(labels_true, labels_pred), 3),
        # 'AMI': round(adjusted_mutual_info_score(labels_true, labels_pred), 3),
        # 'FMI': round(fowlkes_mallows_score(labels_true, labels_pred), 3),
    }

    # Accuracy
    m_labels = set(labels_true)
    n_labels = set(labels_pred)
    cost_matrix = numpy.ndarray((len(m_labels), len(n_labels)))
    for i, u in enumerate(m_labels):
        for j, v in enumerate(n_labels):
            cost_matrix[i][j] = \
                len(set([x for x, _ in enumerate(labels_true) if _ == u]).intersection(
                    [x for x, _ in enumerate(labels_pred) if _ == v]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    quality['ACC'] = round(cost_matrix[row_ind, col_ind].sum() / len(labels_true), 3)

    return quality


if __name__ == '__main__':
    dataset = numpy.asarray([
        (1, 2),
        (1, 2),
        (1, 2),
        (4, 5),
        (5, 6)
    ])
    clusters, _ = cop_kmeans(dataset, 3, ml=[(0, 4), (1, 2), (0, 3)], cl=[(0, 2)])

    print(clusters)
