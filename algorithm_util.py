from pipeline.classification import SimpleClassificationPipeline
from pipeline.regression import SimpleRegressionPipeline

spaces = {
    "classification": SimpleClassificationPipeline,
    "regression": SimpleRegressionPipeline,
}
