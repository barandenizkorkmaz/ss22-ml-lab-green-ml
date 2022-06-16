from base_network import Network

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

class PolynomialRegressor(Network):
    def __init__(self, x, y, degree):
        super().__init__()
        self.x = x
        self.y = y
        self.degree = degree
        self.transformer = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()

    def train(self):
        x_transformed = self.transformer.fit_transform(self.x)
        self.model.fit(x_transformed, self.y)

    def predict(self,x_test):
        x_test_transformed = self.transformer.fit_transform(x_test)
        return self.model.predict(x_test_transformed)

    def compute_loss(self, y_actual, y_predicted, loss_fn):
        return loss_fn(y_actual, y_predicted)