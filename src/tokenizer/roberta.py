from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer


def make_roberta_tokenizer() -> PreTrainedTokenizer:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return tokenizer



