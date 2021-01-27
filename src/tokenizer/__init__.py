from transformers import PreTrainedTokenizer
from src.constants import MODEL_NAME

from src.tokenizer.electra import make_electra_tokenizer
from src.tokenizer.legalbert import make_legalbert_tokenizer
from src.tokenizer.roberta import make_roberta_tokenizer
from src.tokenizer.xlnet import make_xlnet_tokenizer


def create_tokenizer(modelname: str) -> PreTrainedTokenizer:
    """
    Creates the correct tokenizer given the model name
    """

    if modelname in [MODEL_NAME.ROBERTA, MODEL_NAME.ROBERTA_POOLED]:
        return make_roberta_tokenizer()

    elif modelname == MODEL_NAME.ELECTRA:
        return make_electra_tokenizer()

    elif modelname == MODEL_NAME.LEGALBERT:
        return make_legalbert_tokenizer()

    elif modelname == MODEL_NAME.XLNET:
        return make_xlnet_tokenizer()

    else:
        raise NotImplementedError(f"Unknown model '{modelname}'")
