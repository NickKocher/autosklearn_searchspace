from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)

from askl_typing import FEAT_TYPE_TYPE
from pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
from pipeline.constants import DENSE, PREDICTIONS, SPARSE, UNSIGNED_DATA
from pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel,
)
from util.common import check_for_bool, check_none


class ExtraTreesClassifier(
    IterativeComponentWithSampleWeight,
    AutoSklearnClassificationAlgorithm,
):
    def __init__(
        self,
        criterion,
        min_samples_leaf,
        min_samples_split,
        max_features,
        bootstrap,
        max_leaf_nodes,
        max_depth,
        min_weight_fraction_leaf,
        min_impurity_decrease,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        class_weight=None,
    ):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_estimators

    def iterative_fit(self, X, y, sample_weight=None, n_iter=1, refit=False):
        from sklearn.ensemble import ExtraTreesClassifier as ETC

        if refit:
            self.estimator = None

        if self.estimator is None:
            max_features = int(X.shape[1] ** float(self.max_features))
            if self.criterion not in ("gini", "entropy"):
                raise ValueError(
                    "'criterion' is not in ('gini', 'entropy'): " "%s" % self.criterion
                )

            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)

            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_samples_split = int(self.min_samples_split)
            self.max_features = float(self.max_features)
            self.min_impurity_decrease = float(self.min_impurity_decrease)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
            self.oob_score = check_for_bool(self.oob_score)
            self.bootstrap = check_for_bool(self.bootstrap)
            self.n_jobs = int(self.n_jobs)
            self.verbose = int(self.verbose)

            self.estimator = ETC(
                n_estimators=n_iter,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                bootstrap=self.bootstrap,
                max_features=max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                class_weight=self.class_weight,
                warm_start=True,
            )

        else:
            self.estimator.n_estimators += n_iter
            self.estimator.n_estimators = min(
                self.estimator.n_estimators, self.n_estimators
            )

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        return not len(self.estimator.estimators_) < self.n_estimators

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "ET",
            "name": "Extra Trees Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
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

        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini"
        )

        # The maximum number of features used in the forest is calculated as
        # m^max_features, where m is the total number of features,
        # and max_features is the hyperparameter specified below.
        # The default is 0.5, which yields sqrt(m) features as max_features
        # in the estimator. This corresponds with Geurts' heuristic.
        max_features = UniformFloatHyperparameter(
            "max_features", 0.0, 1.0, default_value=0.5
        )

        max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1
        )
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            "min_weight_fraction_leaf", 0.0
        )
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            "min_impurity_decrease", 0.0
        )

        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="False"
        )
        cs.add_hyperparameters(
            [
                criterion,
                max_features,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                min_impurity_decrease,
                bootstrap,
            ]
        )

        return cs
