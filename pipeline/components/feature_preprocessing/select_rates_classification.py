from typing import Optional

from functools import partial

from ConfigSpace import NotEqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

from askl_typing import FEAT_TYPE_TYPE
from pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from pipeline.constants import (
    DENSE,
    INPUT,
    SIGNED_DATA,
    SPARSE,
    UNSIGNED_DATA,
)


class SelectClassificationRates(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, alpha, mode="fpr", score_func="chi2", random_state=None):
        import sklearn.feature_selection

        self.random_state = random_state  # We don't use this
        self.alpha = alpha
        self.mode = mode

        if score_func == "chi2":
            self.score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            self.score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info_classif":
            self.score_func = partial(
                sklearn.feature_selection.mutual_info_classif,
                random_state=self.random_state,
            )
            # mutual info classif constantly crashes without mode percentile
            self.mode = "percentile"
        else:
            raise ValueError(
                "score_func must be in ('chi2, 'f_classif', 'mutual_info_classif') "
                "for classification "
                "but is: %s " % (score_func)
            )

    def fit(self, X, y):
        import scipy.sparse
        import sklearn.feature_selection

        self.alpha = float(self.alpha)

        self.preprocessor = sklearn.feature_selection.GenericUnivariateSelect(
            score_func=self.score_func, param=self.alpha, mode=self.mode
        )

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        import scipy.sparse
        import sklearn.feature_selection

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        if self.preprocessor is None:
            raise NotImplementedError()
        try:
            Xt = self.preprocessor.transform(X)
        except ValueError as e:
            if (
                "zero-size array to reduction operation maximum which has no "
                "identity" in e.message
            ):
                raise ValueError("%s removed all features." % self.__class__.__name__)
            else:
                raise e

        if Xt.shape[1] == 0:
            raise ValueError("%s removed all features." % self.__class__.__name__)
        return Xt

    @staticmethod
    def get_properties(dataset_properties=None):
        data_type = UNSIGNED_DATA

        if dataset_properties is not None:
            signed = dataset_properties.get("signed")
            if signed is not None:
                data_type = SIGNED_DATA if signed is True else UNSIGNED_DATA

        return {
            "shortname": "SR",
            "name": "Univariate Feature Selection based on rates",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, data_type),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1
        )

        if dataset_properties is not None and dataset_properties.get("sparse"):
            choices = ["chi2", "mutual_info_classif"]
        else:
            choices = ["chi2", "f_classif", "mutual_info_classif"]

        score_func = CategoricalHyperparameter(
            name="score_func", choices=choices, default_value="chi2"
        )

        mode = CategoricalHyperparameter("mode", ["fpr", "fdr", "fwe"], "fpr")

        cs = ConfigurationSpace()
        cs.add_hyperparameter(alpha)
        cs.add_hyperparameter(score_func)
        cs.add_hyperparameter(mode)

        # mutual_info_classif constantly crashes if mode is not percentile
        # as a WA, fix the mode for this score
        cond = NotEqualsCondition(mode, score_func, "mutual_info_classif")
        cs.add_condition(cond)

        return cs
