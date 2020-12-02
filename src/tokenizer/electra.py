from transformers import PreTrainedTokenizer
from transformers import ElectraTokenizer


def make_electra_tokenizer() -> PreTrainedTokenizer:
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    return tokenizer


if __name__ == '__main__':

    tokenizer = make_electra_tokenizer()
    print(tokenizer.cls_token)
    print(tokenizer.cls_token_id)

    print(tokenizer.sep_token)
    print(tokenizer.sep_token_id)

