class PipelineError(Exception):
    """Base exception for pipeline-related errors."""


class PipelineInputError(PipelineError):
    """Raised when pipeline inputs are invalid."""


class DataValidationError(PipelineError):
    """Raised when dataset validation fails."""


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""


class EvaluationError(PipelineError):
    """Raised when evaluation fails."""