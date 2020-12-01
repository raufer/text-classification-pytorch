from src import device
from src.models.roberta_classifier import ROBERTAClassifier
from src.utils.checkpoints import load_checkpoint


def load_model(output_path: str) -> ROBERTAClassifier:
    """
    Loads the model in `output_path` and returns it
    in the correct device
    """
    model = ROBERTAClassifier()
    model = model.to(device)

    load_checkpoint(output_path + '/model.pkl', model)
    return model

