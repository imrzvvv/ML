import numpy as np


class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty  # Тип регуляризации, которая применяется к модели
        self.alpha = alpha  # Коэффициент регуляризации
        self.max_iter = max_iter  # Максимальное количество итераций для обучения модели
        self.tol = tol  # Порог сходимости
        self.random_state = random_state  # Зерно для генератора случайных чисел
        self.eta0 = eta0  # Начальная скорость обучения
        self.early_stopping = early_stopping  # Включает раннюю остановку
        self.validation_fraction = validation_fraction  # На валидацию
        self.n_iter_no_change = n_iter_no_change  # Параметр "терпения"
        self.shuffle = shuffle  # Если True, данные перемешиваются перед каждой итерацией обучения
        self.batch_size = batch_size  # Размер мини-батча для градиентного спуска mini-batch
        np.random.seed(random_state)

    def get_penalty_grad(self):
        if self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        elif self.penalty == "l2":
            return 2 * self.alpha * self.coef_

    def fit(self, x, y):
        if self.early_stopping:
            x_validation = x[:int(x.shape[0] * self.validation_fraction)]
            y_validation = y[:int(y.shape[0] * self.validation_fraction)]
            x = x[int(x.shape[0] * self.validation_fraction):]
            y = y[int(y.shape[0] * self.validation_fraction):]

        n_samples = x.shape[0]
        n_features = x.shape[1]
        best_loss = np.inf
        no_improvement_count = 0

        self.coef_ = np.random.randn(n_features)
        self.intercept_ = np.random.randn()

        for _ in range(self.max_iter):
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                x_tmp = x[indices]
                y_tmp = y[indices]
            else:
                x_tmp = x
                y_tmp = y
            for j in range(0, n_samples, self.batch_size):
                x_batch = x_tmp[j:j + self.batch_size]
                y_batch = y_tmp[j:j + self.batch_size]

                y_predicted = np.dot(x_batch, self.coef_) + self.intercept_

                loss_grad = -(2 / len(x_batch)) * (x_batch.T.dot(y_batch - y_predicted))
                grad = loss_grad + self.get_penalty_grad()
                intercept_grad = -(2 / len(x_batch)) * np.sum(y_batch - y_predicted)

                self.coef_ -= self.eta0 * grad
                self.intercept_ -= self.eta0 * intercept_grad

            if self.early_stopping:
                y_predicted = np.dot(x_validation, self.coef_) + self.intercept_
                loss = np.mean(np.square(y_validation - y_predicted))

                if loss < best_loss - self.tol:
                    best_loss = loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count == self.n_iter_no_change:
                        break


    def predict(self, x):
        return np.dot(x, self.coef_) + self.intercept_

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
