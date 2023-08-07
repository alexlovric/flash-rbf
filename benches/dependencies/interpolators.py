import numpy as np

from scipy.interpolate import Rbf


class RBFscipy:
    def __init__(self, x, y, epsilon=None):
        self.x = x
        self.y = y
        self.epsilon = np.sqrt(epsilon)
        # self.rbf = Rbf(x, x, y, function="gaussian", epsilon=self.epsilon)
        self.rbf = Rbf(
            *[x[:, i] for i in range(np.shape(x)[1])],
            y,
            epsilon=self.epsilon,
            function=self.kernel,
            # function="gaussian",
            norm="sqeuclidean",
        )
        # Note the euclidean is used because the default gaussian is squaring

    def predict(self, x_new):
        return self.rbf(*[x_new[:, i] for i in range(np.shape(x_new)[1])])

    def update(self, x_new, y_new):
        self.merge_unique_points(x_new, y_new)
        self.rbf = Rbf(
            *[self.x[:, i] for i in range(np.shape(self.x)[1])],
            self.y,
            epsilon=self.epsilon,
            function=self.kernel,
            # function="gaussian",
            norm="sqeuclidean",
        )

    def merge_unique_points(self, x_new, y_new):
        for point in x_new:
            is_duplicate = np.any(np.all(point == self.x, axis=1))
            if not is_duplicate:
                self.x = np.vstack((self.x, x_new))
                self.y = np.append(self.y, y_new)

    def kernel(self, r):
        r[r < 2.2204460492503131e-12] = 2.2204460492503131e-12
        return np.exp(-0.5 * r / self.epsilon**2)


class RBFnumpy:
    def __init__(self, x, y, sigma=None):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.A = self._rbf_kernel(self.x, self.x)
        self.coef_ = np.linalg.lstsq(self.A, self.y, rcond=None)[0]
        # self.coef_ = lstsq(self.A, self.y, cond=None)[0]

    def _rbf_kernel(self, X, Y):
        dist = cdist(X, Y, "sqeuclidean")
        return np.exp(-0.5 * dist / self.sigma**2)

    def predict(self, x_new):
        A_new = self._rbf_kernel(x_new, self.x)
        return np.dot(A_new, self.coef_)

    def update(self, x_new, y_new):
        self.merge_unique_points(x_new, y_new)
        self.A = self._rbf_kernel(self.x, self.x)
        self.coef_ = np.linalg.lstsq(self.A, self.y, rcond=None)[0]
        # self.coef_ = lstsq(self.A, self.y, cond=None)[0]

    def merge_unique_points(self, x_new, y_new):
        for point in x_new:
            is_duplicate = np.any(np.all(abs(point - self.x) < 1.0e-6, axis=1))
            if not is_duplicate:
                self.x = np.vstack((self.x, x_new))
                self.y = np.append(self.y, y_new)


def cdist(XA, XB, metric="euclidean"):
    """
    Computes the distance between each pair of vectors in XA and XB using the specified metric.
    XA and XB should be 2D arrays, where each row represents a vector.
    """
    m = XA.shape[0]
    n = XB.shape[0]
    d = XA.shape[1]
    assert d == XB.shape[1], "Dimensionality of XA and XB must match."
    dist = np.zeros((m, n))
    if metric == "euclidean":
        for i in range(m):
            for j in range(n):
                dist[i, j] = np.sqrt(np.sum((XA[i] - XB[j]) ** 2))
    elif metric == "cityblock":
        for i in range(m):
            for j in range(n):
                dist[i, j] = np.sum(np.abs(XA[i] - XB[j]))
    elif metric == "sqeuclidean":
        for i in range(m):
            for j in range(n):
                dist[i, j] = np.sum((XA[i] - XB[j]) ** 2)
    elif metric == "cosine":
        for i in range(m):
            for j in range(n):
                dist[i, j] = 1 - np.dot(XA[i], XB[j]) / (
                    np.linalg.norm(XA[i]) * np.linalg.norm(XB[j])
                )
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return dist
