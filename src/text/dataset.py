import logging

from src.constants import MAX_SEQ_LEN

from transformers import PreTrainedTokenizer
from torchtext.data import Field
from torchtext.data import TabularDataset


logger = logging.getLogger(__name__)


def create_dataset(tokenizer: PreTrainedTokenizer, filepath: str, format: str = 'CSV') -> TabularDataset:
    """
    Creates a dataset from the file `filepath`
    Each record contains two fields: 'text', 'label'
    """
    logger.info(f"Creating a dataset from file in '{filepath}'; format '{format}'")

    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    label_field = Field(sequential=False, use_vocab=False, batch_first=True)

    text_field = Field(
        use_vocab=False,
        tokenize=tokenizer.encode,
        include_lengths=False,
        batch_first=True,
        fix_length=MAX_SEQ_LEN,
        pad_token=PAD_INDEX,
        unk_token=UNK_INDEX
    )

    fields = {
        'text': ('text', text_field),
        'label': ('label', label_field)
    }

    dataset = TabularDataset(path=filepath, format=format, fields=fields, skip_header=False)
    return dataset