from transformers import PreTrainedTokenizer
from src.constants import MODEL_NAME

from src.tokenizer.electra import make_electra_tokenizer
from src.tokenizer.legalbert import make_legalbert_tokenizer
from src.tokenizer.roberta import make_roberta_tokenizer


def create_tokenizer(modelname: str) -> PreTrainedTokenizer:
    """
    Creates the correct tokenizer given the model name
    """

    if modelname == MODEL_NAME.ROBERTA:
        return make_roberta_tokenizer()

    elif modelname == MODEL_NAME.ELECTRA:
        return make_electra_tokenizer()

    elif modelname == MODEL_NAME.LEGALBERT:
        return make_legalbert_tokenizer()

    else:
        raise NotImplementedError(f"Unknown model '{modelname}'")
