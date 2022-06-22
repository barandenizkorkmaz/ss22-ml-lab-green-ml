from .base_network import Network

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolynomialRegression(Network):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.transformer = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()

    def train(self, x_train, y_train, x_val, y_val):
        x_transformed = self.transformer.fit_transform(x_train)
        self.model.fit(x_transformed, y_train)

    def predict(self,x_test):
        x_test_transformed = self.transformer.fit_transform(x_test)
        return self.model.predict(x_test_transformed)