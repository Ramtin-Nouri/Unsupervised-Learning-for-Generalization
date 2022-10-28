import argparse
from mimetypes import init
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import json
import os
import dataset
from models.meta_models import EncoderDecoder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json", required=True)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()

def load_config(config_path, debug=False):
    """Load the configuration JSON file.
    If debug is True, the number of samples is reduced.
    If a value is not specified in the JSON file, the default value is used.
    Default values are specified in the configs/default.json file.

    Args:
        config_path (str): Path to the configuration JSON file.
        debug (bool): If True, the number of samples is reduced.
    """
    config_file = json.load(open(config_path, "r"))
    default = json.load(open("configs/default.json", "r"))
    config = dict(
        gpus=config_file.get("gpus", default["gpus"]),
        epochs=config_file.get("epochs", default["epochs"]),
        batch_size=config_file.get("batch_size", default["batch_size"]),
        num_workers=config_file.get("num_workers", default["num_workers"]),
        input_length=config_file.get("input_length", default["input_length"]),
        input_stride=config_file.get("input_stride", default["input_stride"]),
        use_joints=config_file.get("use_joints", default["use_joints"]),
        output_dir=config_file.get("output_dir", default["output_dir"]),
        learning_rate=config_file.get("learning_rate", default["learning_rate"]),
        pretrained = config_file.get("pretrained", default["pretrained"]),
    )
    # dataset related configs
    dataset = config_file.get("dataset", default["dataset"])
    config["data_path"] = dataset.get("data_path", default["dataset"]["data_path"])
    config["num_training_samples"] = dataset.get("num_training_samples", default["dataset"]["num_training_samples"])
    config["num_validation_samples"] = dataset.get("num_validation_samples", default["dataset"]["num_validation_samples"])
    config["visible_objects"] = dataset.get("visible_objects", default["dataset"]["visible_objects"])
    config["different_colors"] = dataset.get("different_colors", default["dataset"]["different_colors"])
    config["different_objects"] = dataset.get("different_objects", default["dataset"]["different_objects"])
    config["exclusive_colors"] = dataset.get("exclusive_colors", default["dataset"]["exclusive_colors"])
    config["different_actions"] = dataset.get("different_actions", default["dataset"]["different_actions"])
    config["num_joints"] = dataset.get("num_joints", default["dataset"]["num_joints"])

    if debug:
        config["num_training_samples"] = 20
        config["num_validation_samples"] = 20
        config["epochs"] = 1
    config["debug"] = debug
    os.makedirs(config["output_dir"], exist_ok=True)

    # Read personal data from different file
    try:
        personal_data = json.load(open("configs/personal_data.json"))
        config["wandb_project"] = personal_data["wandb"]["project"]
        config["wandb_username"] = personal_data["wandb"]["username"]
    except:
        print("Please make sure a JSON file exists called configs/personal_data, \
        including your WandB project name and user name")

    # Add git hash to config
    config["git_hash"] = os.popen("git rev-parse HEAD").read().strip()

    # Path to save the model
    time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = "{}_V{}-A{}-C{}-O{}-{}{}".format(
        time,
        config["visible_objects"],
        config["different_actions"],
        config["different_colors"],
        config["different_objects"],
        "X" if config["exclusive_colors"] else "",
        "-J" if config["use_joints"] else "",
    )
    config["run_name"] = run_name
    config["model_path"] = os.path.join(config["output_dir"], run_name + ".pt")
    
    return config

def main(args):
    pl.seed_everything(42) # for reproducibility
    config = load_config(args.config, args.debug)
    print("Config:", config)

    datamodule = dataset.DataModule(config)

    model = EncoderDecoder(config)
    
    wandb_logger = WandbLogger(
        project=config["wandb_project"],
        name=config["run_name"],
        save_dir=config["output_dir"],
        anonymous="allow",
    )
    wandb_logger.watch(model, log="all")

    # TODO: add debug mode
    trainer = pl.Trainer(
        gpus=config["gpus"],
        max_epochs=config["epochs"],
        logger=wandb_logger,
        log_every_n_steps=1
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    args = parse_args()
    main(args)