from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from sklearn.base import BaseEstimator

from askl_typing import FEAT_TYPE_TYPE
from pipeline.base import DATASET_PROPERTIES_TYPE, BasePipeline
from pipeline.components.data_preprocessing.categorical_encoding import (  # noqa: E501
    OHEChoice,
)
from pipeline.components.data_preprocessing.categorical_encoding.encoding import (  # noqa: E501
    OrdinalEncoding,
)
from pipeline.components.data_preprocessing.category_shift.category_shift import (  # noqa: E501
    CategoryShift,
)
from pipeline.components.data_preprocessing.imputation.categorical_imputation import (  # noqa: E501
    CategoricalImputation,
)
from pipeline.components.data_preprocessing.minority_coalescense import (
    CoalescenseChoice,
)
from pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class CategoricalPreprocessingPipeline(BasePipeline):
    """This class implements a pipeline for data preprocessing of categorical features.
    It assumes that the data to be transformed is made only of categorical features.
    The steps of this pipeline are:
        1 - Category shift: Adds 3 to every category value
        2 - Imputation: Assign category 2 to missing values (NaN).
        3 - Minority coalescence: Assign category 1 to all categories whose occurrence
            don't sum-up to a certain minimum fraction
        4 - One hot encoding: usual sklearn one hot encoding
    Parameters
    ----------
    config : ConfigSpace.configuration_space.Configuration
        The configuration to evaluate.
    random_state : Optional[int | RandomState]
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`."""

    def __init__(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        config: Optional[Configuration] = None,
        steps: Optional[List[Tuple[str, BaseEstimator]]] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        init_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._output_dtype = np.int32
        super().__init__(
            config=config,
            steps=steps,
            dataset_properties=dataset_properties,
            include=include,
            exclude=exclude,
            random_state=random_state,
            init_params=init_params,
            feat_type=feat_type,
        )

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "cat_datapreproc",
            "name": "categorical data preprocessing",
            "handles_missing_values": True,
            "handles_nominal_values": True,
            "handles_numerical_features": True,
            "prefers_data_scaled": False,
            "prefers_data_normalized": False,
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "is_deterministic": True,
            # TODO find out if this is right!
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
            "preferred_dtype": None,
        }

    def _get_hyperparameter_search_space(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        include: Optional[Dict[str, str]] = None,
        exclude: Optional[Dict[str, str]] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        """Create the hyperparameter configuration space.
        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()

        cs = self._get_base_search_space(
            cs=cs,
            feat_type=feat_type,
            dataset_properties=dataset_properties,
            exclude=exclude,
            include=include,
            pipeline=self.steps,
        )

        return cs

    def _get_pipeline_steps(
        self,
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, BaseEstimator]]:
        steps = []

        default_dataset_properties = {}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps = [
            ("imputation", CategoricalImputation(random_state=self.random_state)),
            ("encoding", OrdinalEncoding(random_state=self.random_state)),
            ("category_shift", CategoryShift(random_state=self.random_state)),
            (
                "category_coalescence",
                CoalescenseChoice(
                    feat_type=feat_type,
                    dataset_properties=default_dataset_properties,
                    random_state=self.random_state,
                ),
            ),
            (
                "categorical_encoding",
                OHEChoice(
                    feat_type=feat_type,
                    dataset_properties=default_dataset_properties,
                    random_state=self.random_state,
                ),
            ),
        ]

        return steps

    def _get_estimator_hyperparameter_name(self) -> str:
        return "categorical data preprocessing"
