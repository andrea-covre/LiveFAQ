import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


LOG_CONST = 1e-16


def euclidean_dist(x: np.ndarray, y: np.ndarray, squared=False):
    return euclidean_distances(x, y, squared=squared)


def dot_prod_dist(x: np.ndarray, y: np.ndarray):
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, ord=2, axis=1, keepdims=True)

    norm_mat = x_norm * y_norm.T

    return norm_mat - x @ y.T


def cosine_dist(x: np.ndarray, y: np.ndarray):
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, ord=2, axis=1, keepdims=True)

    x_normalized = x / x_norm
    y_normalized = y / y_norm

    return 1.0 - (x_normalized @ y_normalized.T)


class KMeans(object):
    """
    KMeans class: Implementation based on 7641: Machine Learning HW2
    """
    def __init__(self, num_clusters, dist_fn='', init='k++', n_times=5, max_iters=300, seed=None, tol=1.e-9):
        assert max_iters > 0

        self.num_clusters = num_clusters

        if seed is None:
            self.randomizer = np.random.RandomState()
        else:
            self.randomizer = np.random.RandomState(seed=seed)

        dist_fn = dist_fn.lower()
        assert dist_fn in ['', 'euclidean', 'dot_prod', 'inner_prod', 'cosine']

        if dist_fn in ['', 'euclidean']:
            dist_fn = euclidean_dist
            self.dist_name = 'euclidean'
        elif dist_fn in ['dot_prod', 'inner_prod']:
            dist_fn = dot_prod_dist
            self.dist_name = 'dot_prod'
        elif dist_fn in ['cosine']:
            dist_fn = cosine_dist
            self.dist_name = 'cosine'
        else:
            raise ValueError

        init = init.lower()

        if init in ['k++', 'kmeans++']:
            init = 'k++'
        elif init in ['rand', 'random']:
            init = 'random'
        else:
            raise ValueError

        self.init = init

        self.dist_fn = dist_fn

        self.tol = tol
        self.rel_tol = 1.e-5

        self.n_times = n_times

        self.max_iters = max_iters

        self.centers = None

        self.cluster_labels = None

    def _init_centers_random(self, points: np.ndarray):
        num_pts = points.shape[0]

        max_r = max(num_pts, self.num_clusters)

        x = self.randomizer.choice(np.arange(max_r), size=self.num_clusters, replace=False) % num_pts

        self.centers = points[x]

    def _init_centers_kpp(self, points: np.ndarray):
        """
        Code based on sklearn KMeans: KMeans++ initialization
        :param points: Input array of points to be clustered
        :return: None
        """
        assert points.ndim == 2
        num_points = points.shape[0]

        n_local_trials = 2 + int(np.log(self.num_clusters))

        centers = np.zeros((self.num_clusters, points.shape[1]), dtype=float)

        center_idx = self.randomizer.randint(num_points)

        centers[0] = points[center_idx]

        if self.dist_name == 'euclidean':
            dist_fn = lambda x, y: euclidean_dist(x, y, squared=True)
        else:
            dist_fn = lambda x, y: cosine_dist(x, y)

        closest_dist = dist_fn(centers[0][None, ...], points)

        current_pot = closest_dist.sum()

        for c_i in range(1, self.num_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = self.randomizer.uniform(size=n_local_trials) * current_pot

            candidate_ids = np.searchsorted(np.cumsum(closest_dist), rand_vals)
            np.clip(candidate_ids, None, closest_dist.size - 1, out=candidate_ids)

            dist_to_candidates = dist_fn(points[candidate_ids], points)

            np.minimum(closest_dist, dist_to_candidates, out=dist_to_candidates)

            candidate_pots = dist_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidate_pots)
            current_pot = candidate_pots[best_candidate]
            closest_dist = dist_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            centers[c_i] = points[best_candidate]

        self.centers = centers

    def _update_assignment(self, points: np.ndarray):
        distance_mat = self.dist_fn(x=points, y=self.centers)
        return np.argmin(distance_mat, axis=1)

    def _update_centers(self, points: np.ndarray, cluster_idx: np.ndarray):
        new_centers = np.copy(self.centers)

        for c_i in range(self.num_clusters):
            points_in_cluster = points[cluster_idx == c_i]
            num_points = points_in_cluster.shape[0]

            if num_points > 0:
                new_centers[c_i] = np.mean(points_in_cluster, axis=0)

        self.centers = new_centers

    def _get_loss(self, points: np.ndarray, cluster_idx: np.ndarray):
        if self.dist_name == 'euclidean':
            dist_fn = lambda x, y: euclidean_dist(x, y, squared=True)
        else:
            dist_fn = lambda x, y: cosine_dist(x, y)

        dist_sum = 0

        for c_i in range(self.num_clusters):
            dist_from_center = dist_fn(points[cluster_idx == c_i], self.centers[c_i][None, :])
            dist_sum += np.abs(dist_from_center).sum()

        return dist_sum / points.shape[0]

    def _fit_one_time(self, x:np.ndarray):
        if self.init == 'k++':
            self._init_centers_kpp(x)
        elif self.init == 'random':
            self._init_centers_random(x)
        else:
            raise ValueError

        loss = 0

        for it in range(self.max_iters):
            cluster_idx = self._update_assignment(x)
            self._update_centers(points=x, cluster_idx=cluster_idx)
            loss = self._get_loss(x, cluster_idx)

            self.cluster_labels = cluster_idx

            if it:
                diff = np.abs(prev_loss - loss)
                if diff < self.tol and diff / prev_loss < self.rel_tol:
                    break

            prev_loss = loss

        return loss

    def fit(self, x: np.ndarray):
        assert x.ndim == 2

        centers_list = list()
        c_labels_list = list()
        loss_list = list()

        for i in range(self.n_times):
            loss = self._fit_one_time(x)

            loss_list.append((loss, i))
            centers_list.append(np.copy(self.centers))
            c_labels_list.append(np.copy(self.cluster_labels))

        loss_list.sort()
        self.centers = centers_list[loss_list[0][1]]
        self.cluster_labels = c_labels_list[loss_list[0][1]]

        return self

    def predict(self, x: np.ndarray):
        assert self.centers is not None
        assert x.ndim == 2

        return self._update_assignment(x)


def softmax(logit):
    max_logit = np.max(logit, axis=1, keepdims=True)

    exp_logit = np.exp(logit - max_logit)

    exp_logit /= np.sum(exp_logit, axis=1, keepdims=True)

    return exp_logit


def logsumexp(logit):
    max_logits = np.max(logit, axis=1, keepdims=False)

    exp_logit = np.exp(logit - max_logits[:, None])
    log_sum_exp = np.log(exp_logit.sum(axis=1) + LOG_CONST)
    log_sum_exp += max_logits

    return log_sum_exp[:, None]

