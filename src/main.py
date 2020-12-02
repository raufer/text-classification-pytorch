import torch
import json
import logging

from torch.utils.data import DataLoader

from src import device
from src.ops.evaluation import evaluate
from src.ops.loss import cross_entropy_loss
from src.ops.metrics import classification_report, write_confusion_matrix
from src.processes.pretrain import pretrain
from src.processes.train import train
from src.utils.directories import make_run_dir
from src.viz.metrics import write_train_valid_loss

from src.ops.weights import calculate_multiclass_weights
from src.data.dataset import create_datasets
from src.data.iterator import make_iterators

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from typing import Dict
from typing import List


logger = logging.getLogger(__name__)


def training_job(config: Dict, model: torch.nn.Module, train_iter: DataLoader, valid_iter: DataLoader, test_iter: DataLoader, output_dir: str, weights: List = None):
    """
    Main training loop
    """
    logger.info(f"Text Classification Training Pipeline")
    logger.info("Configuration")
    logger.info(json.dumps(config, indent=4))

    output_path = make_run_dir(output_dir)

    model = model.to(device)

    steps_per_epoch = len(train_iter)

    loss_function = cross_entropy_loss(weights)

    lr = config['learning-rate-pretrain']
    n_epochs = config['num-epochs-pretrain']
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*1, num_training_steps=steps_per_epoch*n_epochs)

    pretrain(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        valid_period=len(train_iter),
        num_epochs=n_epochs
    )

    lr = config['learning-rate-train']
    n_epochs = config['num-epochs-train']
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch * 2, num_training_steps=steps_per_epoch * n_epochs)
    train(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        valid_period=len(train_iter)/3,
        num_epochs=n_epochs,
        output_path=output_path
    )

    write_train_valid_loss(output_path)

    # model = load_model(output_path)
    y_true, y_pred = evaluate(model, test_iter)

    classification_report(y_true, y_pred)
    write_confusion_matrix(y_true, y_pred, output_path)


if __name__ == '__main__':

    import pandas as pd

    from src.config import config
    from src.arguments import args

    from src.models import make_model
    from src.tokenizer import create_tokenizer

    datapath = args.data_path
    output_dir = args.output_dir
    modelname = args.model

    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: '{value}'")

    df = pd.read_csv(datapath)
    n_outputs = df['label'].nunique()

    tokenizer = create_tokenizer(modelname)

    model = make_model(modelname)(
        dropout_rate=config['dropout-ratio'],
        n_outputs=n_outputs
    )

    split_ratios = [0.7, 0.2, 0.1]
    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer=tokenizer, filepath=datapath, split_ratios=split_ratios)
    train_iter, valid_iter, test_iter = make_iterators(train_dataset, val_dataset, test_dataset)

    weights = calculate_multiclass_weights(df['label'])

    training_job(
        config=config,
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        test_iter=test_iter,
        output_dir=output_dir,
        weights=weights
    )









