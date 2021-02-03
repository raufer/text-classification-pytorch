import os
import logging
from itertools import islice
from itertools import repeat
from collections import defaultdict

from src.batch.elements.source import traverse_documents
from src.batch.elements.transform import process_single, run_inference

logger = logging.getLogger(__name__)


def pipeline(input_dir: str, output_dir: str, model_dir: str, n_outputs: int):

    logger.info("Running main pipeline")
    logger.info(f"Input dir '{input_dir}'")
    logger.info(f"Output dir '{output_dir}'")
    logger.info(f"Model dir '{model_dir}'")

    stream = traverse_documents(input_dir)
    stream = list((islice(stream, 10)))

    stream = (xs + (process_single(*xs),) for xs in stream)

    stream = (list(zip(repeat(file), xs)) for file, doc, g, xs in stream)

    data = sum(stream, [])
    data = [(file,) + xs for file, xs in data]

    texts = [os.linesep.join(x[-1]) + os.linesep for x in data]

    # y_pred, y_probs = run_inference(texts, model_dir, n_outputs)

    import pickle

    with open('y_pred.pkl', 'rb') as f:
        y_pred = pickle.load(f)

    with open('y_probs.pkl', 'rb') as f:
        y_probs = pickle.load(f)

    documents = defaultdict(lambda : {'articles': []})

    for i in data:
        print(i)
    raise ValueError

    return stream


if __name__ == '__main__':

    from src.batch.arguments import args

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    n_outputs = args.n_outputs

    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: '{value}'")

    # input_dir = '/Users/raulferreira/waymark/data/prepared/enacted-epublished-xml'
    # output_dir = '/Users/raulferreira/waymark/data/nlp-outputs/regulatory-burden'
    # model_dir = ''
    # n_outputs = 4

    pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        model_dir=model_dir,
        n_outputs=n_outputs
    )
