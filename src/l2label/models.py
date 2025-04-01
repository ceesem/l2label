import glob
import json
import pathlib
from typing import Optional

model_dir = pathlib.Path(__file__).parent / "models"

current_model = "model_ds15_us0_bd0.json"


def get_models(
    dir: Optional[str] = None,
):
    """
    Get a list of all models in the model directory.
    """
    if dir is None:
        dir = model_dir

    return [f.name for f in glob.glob(f"{dir}/*.json")]


def load_model_config(
    model_name: Optional[str] = None,
    dir: Optional[str] = None,
):
    """
    Load a model configuration file.

    Parameters
    ----------
    model_name : str
        The filename name of the model to load.
    dir : str, optional
        The directory to load the model from. If None, the default model directory is used.
    """
    if dir is None:
        dir = model_dir
        if model_name is None:
            model_name = current_model

    if model_name is None:
        raise ValueError("model_name must be specified if dir is not provided")
    if not model_name.endswith(".json"):
        model_name += ".json"
    with open(pathlib.Path(dir) / f"{model_name}", "r") as f:
        model_config = json.load(f)

    model_config["model_file"] = pathlib.Path(dir) / f"{model_config['model_file']}"
    return model_config
