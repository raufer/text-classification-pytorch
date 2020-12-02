from src.constants import MODEL_NAME

from src.models.electra_classifier import ElectraClassifier
from src.models.legalbert_classifier import LegalBertClassifier
from src.models.roberta_classifier import ROBERTAClassifier


def make_model(modelname: str):
    """
    Returns the correct model based on the selection
    """
    if modelname == MODEL_NAME.ROBERTA:
        return ROBERTAClassifier

    elif modelname == MODEL_NAME.ELECTRA:
        return ElectraClassifier

    elif modelname == MODEL_NAME.LEGALBERT:
        return LegalBertClassifier

    else:
        raise NotImplementedError(f"Unknown model '{modelname}'")
