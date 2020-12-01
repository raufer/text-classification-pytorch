import logging
import torch

from src.utils.checkpoints import save_checkpoint
from src.utils.metrics import save_metrics
from src.text.tokenizer import PAD_INDEX


logger = logging.getLogger(__name__)


def train(model, optimizer, train_iter, valid_iter, valid_period, output_path, loss_function, scheduler=None, num_epochs=5):
    """
    Training phase
    """

    train_loss = 0.0
    valid_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('Inf')

    global_step = 0
    global_steps_list = []

    model.train()

    for epoch in range(num_epochs):
        for (source, target), _ in train_iter:

            mask = (source != PAD_INDEX).type(torch.uint8)

            y_pred = model(input_ids=source, attention_mask=mask)

            loss = loss_function(y_pred, target)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1

            if global_step % valid_period == 0:
                model.eval()

                with torch.no_grad():
                    for (source, target), _ in valid_iter:
                        mask = (source != PAD_INDEX).type(torch.uint8)

                        y_pred = model(input_ids=source, attention_mask=mask)

                        loss = loss_function(y_pred, target)

                        valid_loss += loss.item()

                train_loss = train_loss / valid_period
                valid_loss = valid_loss / len(valid_iter)
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
                global_steps_list.append(global_step)

                logger.info('Epoch [{}/{}], global step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_iter),
                              train_loss, valid_loss))

                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    save_checkpoint(output_path + '/model.pkl', model, best_valid_loss)
                    save_metrics(output_path + '/metric.pkl', train_loss_list, valid_loss_list, global_steps_list)

                train_loss = 0.0
                valid_loss = 0.0
                model.train()

    save_metrics(output_path + '/metric.pkl', train_loss_list, valid_loss_list, global_steps_list)
    logger.info('Training done!')