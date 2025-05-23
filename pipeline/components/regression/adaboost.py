from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from askl_typing import FEAT_TYPE_TYPE
from pipeline.components.base import AutoSklearnRegressionAlgorithm
from pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA


class AdaboostRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_estimators, learning_rate, loss, max_depth, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, y):
        import sklearn.ensemble
        import sklearn.tree

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = sklearn.tree.DecisionTreeRegressor(max_depth=self.max_depth)

        self.estimator = sklearn.ensemble.AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=self.random_state,
        )

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "AB",
            "name": "AdaBoost Regressor",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        # base_estimator = Constant(name="base_estimator", value="None")
        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True
        )
        loss = CategoricalHyperparameter(
            name="loss",
            choices=["linear", "square", "exponential"],
            default_value="linear",
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=1, log=False
        )

        cs.add_hyperparameters([n_estimators, learning_rate, loss, max_depth])
        return cs
