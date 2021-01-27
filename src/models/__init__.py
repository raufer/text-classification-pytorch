from src.constants import MODEL_NAME

from src.models.electra_classifier import ElectraClassifier
from src.models.legalbert_classifier import LegalBertClassifier
from src.models.roberta_pooled_classifier import RobertaPooledClassifier
from src.models.roberta_classifier import RobertaClassifier
from src.models.xlnet_classifier import XLNETClassifier


def make_model(modelname: str):
    """
    Returns the correct model based on the selection
    """
    if modelname == MODEL_NAME.ROBERTA:
        return RobertaClassifier

    elif modelname == MODEL_NAME.ROBERTA_POOLED:
        return RobertaPooledClassifier

    elif modelname == MODEL_NAME.ELECTRA:
        return ElectraClassifier

    elif modelname == MODEL_NAME.LEGALBERT:
        return LegalBertClassifier

    elif modelname == MODEL_NAME.XLNET:
        return XLNETClassifier

    else:
        raise NotImplementedError(f"Unknown model '{modelname}'")
