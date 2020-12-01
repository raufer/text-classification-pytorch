import json
import logging

from src import device
from src.config import config
from src.io.model import load_model
from src.models.roberta_classifier import ROBERTAClassifier
from src.ops.evaluation import evaluate
from src.ops.loss import cross_entropy_loss
from src.ops.metrics import classification_report
from src.processes.pretrain import pretrain
from src.processes.train import train
from src.utils.directories import make_run_dir
from src.viz.metrics import write_train_valid_loss

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from torchtext.data import Iterator

from typing import Dict
from typing import List


logger = logging.getLogger(__name__)


def _pre_training_loop(model, train_iter, valid_iter, optimizer, scheduler, loss_function, n_epochs):

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


def _training_loop(model, train_iter, valid_iter, optimizer, scheduler, loss_function, n_epochs, output_path):
    train(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        valid_period=len(train_iter),
        num_epochs=n_epochs,
        output_path=output_path
    )


def training_job(config: Dict, train_iter: Iterator, valid_iter: Iterator, test_iter: Iterator, output_dir: str, weights: List = None):
    """
    Main training loop
    """
    logger.info(f"Text Classification Training Pipeline")
    logger.info("Configuration")
    logger.info(json.dumps(config, indent=4))

    output_path = make_run_dir(output_dir)

    model = ROBERTAClassifier(dropout_rate=config['dropout-ratio'])
    model = model.to(device)

    steps_per_epoch = len(train_iter)

    loss_function = cross_entropy_loss(weights)

    lr = config['learning-rate-pretrain']
    n_epochs = config['num-epochs-pretrain']
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch*1, num_training_steps=steps_per_epoch*n_epochs)
    _pre_training_loop(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,

        n_epochs=n_epochs
    )

    lr = config['learning-rate-train']
    n_epochs = config['num-epochs-train']
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch * 2, num_training_steps=steps_per_epoch * n_epochs)
    _training_loop(
        model=model,
        train_iter=train_iter,
        valid_iter=valid_iter,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        output_path=output_path,
        n_epochs=n_epochs
    )

    write_train_valid_loss(output_path)

    model = load_model(output_path)
    y_true, y_pred = evaluate(model, test_iter)

    classification_report(y_true, y_pred)


if __name__ == '__main__':
    ...









