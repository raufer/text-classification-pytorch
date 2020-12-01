from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer


def make_tokenizer() -> PreTrainedTokenizer:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return tokenizer


tokenizer = make_tokenizer()

PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
