import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size
        np.random.seed(self.random_state)

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        y = np.eye(n_classes)[y]

        if self.early_stopping:
            X_validation = X[:int(X.shape[0] * self.validation_fraction)]
            y_validation = y[:int(y.shape[0] * self.validation_fraction)]
            X = X[int(X.shape[0] * self.validation_fraction):]
            y = y[int(y.shape[0] * self.validation_fraction):]

        n_samples, n_features = X.shape
        best_loss = np.inf
        no_improvement_count = 0

        self.intercept_ = np.random.randn(n_classes)
        self.coef_ = np.random.randn(n_features, n_classes)

        for _ in range(self.max_iter):
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                x_tmp = X[indices]
                y_tmp = y[indices]
            else:
                x_tmp = X
                y_tmp = y
            for j in range(0, n_samples, self.batch_size):
                x_batch = x_tmp[j:j + self.batch_size]
                y_batch = y_tmp[j:j + self.batch_size]
                y_predicted = self.predict_proba(x_batch)

                loss_grad = (1 / len(x_batch)) * (x_batch.T @ (y_predicted - y_batch))
                grad = loss_grad + self.get_penalty_grad()
                intercept_grad = (1 / len(x_batch)) * np.sum(y_predicted - y_batch, axis=0)

                self.intercept_ -= self.eta0 * intercept_grad
                self.coef_ -= self.eta0 * grad

            if self.early_stopping:
                y_predicted = self.predict_proba(X_validation)
                loss = -(1 / self.validation_fraction) * np.trace(y_validation.T @ np.log(y_predicted))

                if loss < best_loss - self.tol:
                    best_loss = loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count == self.n_iter_no_change:
                        break

    def get_penalty_grad(self):
        if self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        elif self.penalty == "l2":
            return 2 * self.alpha * self.coef_

    def predict_proba(self, X):
        logits = X @ self.coef_ + self.intercept_
        return self.softmax(logits)

    def predict(self, X):
        y_predicted = self.predict_proba(X)
        return np.argmax(y_predicted, axis=1)

    @staticmethod
    def softmax(z):
        if len(z.shape) == 1:
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @property
    def coef_(self):
        return self.coef

    @property
    def intercept_(self):
        return self.intercept

    @coef_.setter
    def coef_(self, value):
        self.coef = value

    @intercept_.setter
    def intercept_(self, value):
        self.intercept = value
