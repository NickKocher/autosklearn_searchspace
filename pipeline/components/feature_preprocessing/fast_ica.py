from typing import Optional

import warnings

from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

from askl_typing import FEAT_TYPE_TYPE
from pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from pipeline.constants import DENSE, INPUT, UNSIGNED_DATA
from util.common import check_none


class FastICA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, algorithm, whiten, fun, n_components=None, random_state=None):
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.n_components = n_components

        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.decomposition

        # self.whiten = check_for_bool(self.whiten)
        if self.whiten == "False":
            self.whiten = False
        if check_none(self.n_components):
            self.n_components = None
        else:
            self.n_components = int(self.n_components)

        self.preprocessor = sklearn.decomposition.FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            fun=self.fun,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        # Make the RuntimeWarning an Exception!
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="array must not contain infs or NaNs"
            )
            try:
                self.preprocessor.fit(X)
            except ValueError as e:
                if "array must not contain infs or NaNs" in e.args[0]:
                    raise ValueError(
                        "Bug in scikit-learn: "
                        "https://github.com/scikit-learn/scikit-learn/pull/2738"
                    )

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "FastICA",
            "name": "Fast Independent Component Analysis",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": False,
            "input": (DENSE, UNSIGNED_DATA),
            "output": (INPUT, UNSIGNED_DATA),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()

        n_components = UniformIntegerHyperparameter(
            "n_components", 10, 2000, default_value=100
        )
        algorithm = CategoricalHyperparameter(
            "algorithm", ["parallel", "deflation"], "parallel"
        )
        whiten = CategoricalHyperparameter(
            "whiten", ["False", "unit-variance", "arbitrary-variance"], "False"
        )
        fun = CategoricalHyperparameter("fun", ["logcosh", "exp", "cube"], "logcosh")
        cs.add_hyperparameters([n_components, algorithm, whiten, fun])

        cs.add_condition(
            InCondition(n_components, whiten, ["unit-variance", "arbitrary-variance"])
        )
        return cs
