import os

import numpy as np

import torch
import logging
import warnings


warnings.filterwarnings("ignore")

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

logging.basicConfig(
    format='%(asctime)s, %(name)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

torch.manual_seed(4)
# np.random.seed(2)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')