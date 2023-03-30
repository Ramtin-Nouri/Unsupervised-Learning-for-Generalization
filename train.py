import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LambdaCallback
from pytorch_lightning.callbacks import EarlyStopping
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
import json
import os
import dataset_multimodal, dataset_arcgen, dataset_cater
from models.classifier_model import LstmClassifier
from models.lstm_autoencoder import LstmAutoencoder
from helper import *
from evaluation import *

def parse_args():
    """Parse command line arguments.
    Namely the config file and debug mode.
    Config file is a json file with all the parameters.
    Debug mode is a boolean that determines if the code is run in debug mode. By default it is False.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json", required=True)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--no-log", type=bool, default=False) # if True, no wandb logging is done
    parser.add_argument("--mode", type=str, default="both", required=True, help="supervised, unsupervised or both")
    parser.add_argument("--unsupervised_model", type=str, default=None, help="path to unsupervised model. Has to be specified if mode is supervised")

    return parser.parse_args()

def load_config(config_path, debug=False):
    """Load the configuration JSON file.
    If debug is True, the number of samples is reduced.
    If a value is not specified in the JSON file, the default value is used.
    Default values are specified in the configs/default.json file.

    Args:
        config_path (str): Path to the configuration JSON file.
        debug (bool): If True, the number of samples is reduced.

    Returns:
        dict: Configuration dictionary.
    """
    config_file = json.load(open(config_path, "r"))
    default = json.load(open("configs/default.json", "r"))
    config = dict(
        gpus=config_file.get("gpus", default["gpus"]),
        epochs=config_file.get("epochs", default["epochs"]),
        unsupervised_epochs=config_file.get("unsupervised_epochs", default["unsupervised_epochs"]),
        batch_size=config_file.get("batch_size", default["batch_size"]),
        num_workers=config_file.get("num_workers", default["num_workers"]),
        input_length=config_file.get("input_length", default["input_length"]),
        init_length=config_file.get("init_length", default["init_length"]),
        predict_ahead=config_file.get("predict_ahead", default["predict_ahead"]),
        input_stride=config_file.get("input_stride", default["input_stride"]),
        use_joints=config_file.get("use_joints", default["use_joints"]),
        output_dir=config_file.get("output_dir", default["output_dir"]),
        learning_rate=config_file.get("learning_rate", default["learning_rate"]),
        early_stopping_patience=config_file.get("early_stopping_patience", default["early_stopping_patience"]),
        data_augmentation=config_file.get("data_augmentation", default["data_augmentation"]),
    )

    # model architecture related configs
    model = config_file.get("model", default["model"])
    config["convlstm_layers"] = model.get("convlstm_layers", default["model"]["convlstm_layers"])
    config["convolution_layers_decoder"] = model.get("convolution_layers_decoder", default["model"]["convolution_layers_decoder"])
    config["lstm_num_layers"] = model.get("lstm_num_layers", default["model"]["lstm_num_layers"])
    config["lstm_hidden_size"] = model.get("lstm_hidden_size", default["model"]["lstm_hidden_size"])
    config["dropout_autoencoder"] = model.get("dropout_autoencoder", default["model"]["dropout_autoencoder"])
    config["dropout_classifier"] = model.get("dropout_classifier", default["model"]["dropout_classifier"])
    config["use_resnet"] = model.get("use_resnet", default["model"]["use_resnet"])
    config["use_mask"] = model.get("use_mask", default["model"]["use_mask"])

    # dataset related configs
    dataset = config_file.get("dataset", default["dataset"])
    config["dataset_name"] = dataset.get("dataset_name", default["dataset"]["dataset_name"]) # Multimodal or ARC-GEN
    config["width"] = dataset.get("width", default["dataset"]["width"])
    config["height"] = dataset.get("height", default["dataset"]["height"])
    config["data_path"] = dataset.get("data_path", default["dataset"]["data_path"])
    config["num_training_samples"] = dataset.get("num_training_samples", default["dataset"]["num_training_samples"])
    config["num_validation_samples"] = dataset.get("num_validation_samples", default["dataset"]["num_validation_samples"])
    config["num_training_samples_unsupervised"] = dataset.get("num_training_samples_unsupervised", default["dataset"]["num_training_samples_unsupervised"])
    config["num_validation_samples_unsupervised"] = dataset.get("num_validation_samples_unsupervised", default["dataset"]["num_validation_samples_unsupervised"])
    config["visible_objects"] = dataset.get("visible_objects", default["dataset"]["visible_objects"])
    config["different_colors"] = dataset.get("different_colors", default["dataset"]["different_colors"])
    config["different_objects"] = dataset.get("different_objects", default["dataset"]["different_objects"])
    config["exclusive_colors"] = dataset.get("exclusive_colors", default["dataset"]["exclusive_colors"])
    config["different_actions"] = dataset.get("different_actions", default["dataset"]["different_actions"])
    config["num_joints"] = dataset.get("num_joints", default["dataset"]["num_joints"])
    config["sentence_length"] = dataset.get("sentence_length", default["dataset"]["sentence_length"])
    config["dictionary_size"] = dataset.get("dictionary_size", default["dataset"]["dictionary_size"])
    config["multi_sentence"] = dataset.get("multi_sentence", default["dataset"]["multi_sentence"])

    if debug:
        config["num_training_samples"] = 10
        config["num_validation_samples"] = 10
        config["num_training_samples_unsupervised"] = 5
        config["num_validation_samples_unsupervised"] = 5
        config["epochs"] = 1
        config["unsupervised_epochs"] = 1
    config["debug"] = debug
    os.makedirs(config["output_dir"], exist_ok=True)

    # wandb related configs
    wandb = config_file.get("wandb", default["wandb"])
    config["wandb_project"] = wandb.get("project", default["wandb"]["project"])
    config["wandb_username"] = wandb.get("username", default["wandb"]["username"])

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
    config["model_path"] = os.path.join(config["output_dir"], run_name)
    os.makedirs(config["model_path"], exist_ok=True)
    
    return config


def predict_train_val_images(datamodule, model, logger, config):
    """Predicts the images in the training and validation set and saves them to the logger.
    Revert the normalization of the images before saving them.
    Clamp the values to be between 0 and 1 s.t. the visialization is more clear.

    Args:
        datamodule (DataModule): The datamodule to use.
        model (Model): The model to use.
        logger (Logger): The logger to use.
        config (dict): The config to use.
    """
    with torch.no_grad():
        dataset_mean = [0.7605, 0.7042, 0.6045]
        dataset_std = [0.1832, 0.2083, 0.2902]
        unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(dataset_mean, dataset_std)],
            std=[1 / s for s in dataset_std],
        )

        predict_batches = 3
        # predict on train images
        train_iter = iter(datamodule.train_dataloader())
        for _ in range(predict_batches):
            train_batch = next(train_iter)
            train_batch = [x.to("cuda" if torch.cuda.is_available() else "cpu") for x in train_batch]#TODO: make configurable?
            pred = model.predict(train_batch)
            if config["use_joints"]:
                pred = pred[0]
            pred = unnormalize(pred)
            pred = torch.clamp(pred, 0, 1)
            target = train_batch[0][:,-1]
            logger.log_image(key="train", images=[pred, target], caption=["prediction", "target"]) 

        # predict on val images
        val_iter = iter(datamodule.val_dataloader())
        for _ in range(predict_batches):
            val_batch = next(val_iter)
            val_batch = [x.to("cuda" if torch.cuda.is_available() else "cpu") for x in val_batch]#TODO: make configurable?
            pred = model.predict(val_batch)
            if config["use_joints"]:
                pred = pred[0]
            pred = unnormalize(pred)
            pred = torch.clamp(pred, 0, 1)
            target = val_batch[0][:,-1]
            logger.log_image(key="val", images=[pred, target], caption=["prediction", "target"])    
    

def train_unsupervised(config, wandb_logger):
    """Train the unsupervised model.

    Args:
        config (dict): Configuration dictionary.
        wandb_logger (WandbLogger): WandB logger.

    Returns:
        LstmAutoencoder: Trained unsupervised model.
    """
    if config["dataset_name"] == "Multimodal":
        unsupervised_datamodule = dataset_multimodal.DataModule(config, True)
    else:
        unsupervised_datamodule = dataset_cater.CaterModule(config)
    unsupervised_model = LstmAutoencoder(config)
    print("Unsupervised model:", unsupervised_model)
    if wandb_logger is not None:
        wandb_logger.watch(unsupervised_model, log="all")

    unsupervised_checkpt = pl.callbacks.ModelCheckpoint(
        dirpath=config["model_path"],
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
        filename='unsupervised_{epoch}-{val_loss:.3f}'
    )

    lambda_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: 
                                     predict_train_val_images(unsupervised_datamodule, unsupervised_model, wandb_logger, config))

    early_stopping = EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"], mode="min")

    callbacks = [unsupervised_checkpt, lambda_callback, early_stopping] if not config["debug"] else [lambda_callback]

    unsupervised_trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["gpus"],
        max_epochs=config["unsupervised_epochs"],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=1
    )
    unsupervised_trainer.fit(unsupervised_model, datamodule=unsupervised_datamodule)
    with torch.no_grad():
        # Log test images
        i = 0
        for batch in tqdm(unsupervised_datamodule.test_dataloader(), desc="Test Unsupervised"): #batch_size = 1
            pred = unsupervised_model.predict(batch)
            if config["use_joints"]:
                pred = pred[0]
            target = batch[0][0][-1]
            if wandb_logger is not None:
                wandb_logger.log_image(key=f"test_image", step=i, images=[pred, target], caption=["prediction", "target"])
            i += 1

    if config["debug"]:
        return unsupervised_model
    best = load_model(unsupervised_checkpt.best_model_path, is_unsupervised=True)
    return best

def load_model(model_path, is_unsupervised, encoder=None):
    """Load the model.

    Args:
        model_path (str): Path to the model.
        is_unsupervised (bool): Whether the model is unsupervised.

    Returns:
        LstmClassifier or LstmAutoencoder: Loaded model.
    """
    # Load model
    model_ckpt = torch.load(model_path)
    saved_config = model_ckpt["hyper_parameters"]["config"]

    if is_unsupervised:
        model = LstmAutoencoder(saved_config)
    else:
        model = LstmClassifier(saved_config, encoder)
    model.load_state_dict(model_ckpt["state_dict"])
    return model

def train_supervised(config, wandb_logger, encoder=None):
    """Train the supervised model.

    Args:
        config (dict): Configuration dictionary.
        wandb_logger (WandbLogger): WandB logger.
        encoder (LstmAutoencoder): Trained unsupervised model.

    Returns:
        LstmAutoencoder: Trained supervised model.
        datamodule (DataModule): Supervised datamodule.
    """
    if config["dataset_name"] == "Multimodal":
        supervised_datamodule = dataset_multimodal.DataModule(config, False)
    else:
        supervised_datamodule = dataset_arcgen.DataModule(config)
    supervised_model = LstmClassifier(config, encoder)
    print("Supervised model:", supervised_model)
    if wandb_logger is not None:
        wandb_logger.watch(supervised_model, log="all")

    supervised_checkpt = pl.callbacks.ModelCheckpoint(
        dirpath=config["model_path"],
        save_top_k=1,
        verbose=True,
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        filename='supervised_{epoch}-{val_loss:.3f}'
    )

    early_stopping = EarlyStopping(monitor="val_loss/dataloader_idx_0", patience=config["early_stopping_patience"], mode="min")
    callbacks = [supervised_checkpt, early_stopping] if not config["debug"] else []

    supervised_trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["gpus"],
        max_epochs=config["epochs"],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=1
    )
    supervised_trainer.fit(supervised_model, datamodule=supervised_datamodule)

    if config["debug"]:
        return supervised_model, supervised_datamodule
    best = load_model(supervised_checkpt.best_model_path, is_unsupervised=False, encoder=encoder)
    return best,supervised_datamodule

def evaluate(config, wandb_logger, model, dataloader, name):
    """Evaluate the model on the given dataloader.
    Creates a confusion matrix and logs it to wandb.

    Args:
        config (dict): Configuration dictionary.
        wandb_logger (WandbLogger): WandB logger.
        model (LstmClassifier): Trained model.
        dataloader (DataLoader): Dataloader to evaluate on.
        name (str): Name of the dataloader.

    Returns:
        float: The sentence-wise accuracy.
    """
    # ugly compatibility stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO: avoid passing device, use pytorch lightning
    model.to(device)
    if config["dataset_name"] == "Multimodal":
        dictionary = ["put down", "picked up", "pushed left", "pushed right",
                  "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                  "red", "green", "blue", "yellow", "white", "brown"]
    else:
        dictionary = ['EOS', '_containing', '_contain', '_pick_place', '_rotate', '_slide',
              'metal', 'rubber',
              'yellow', 'cyan', 'gold', 'brown', 'red', 'gray', 'purple', 'blue', 'green',
              'sphere', 'cube', 'cylinder', 'cone', 'spl']

    confusion_matrix_absolute, sentence_wise_accuracy = get_evaluation(
        model,
        dataloader,
        device,
        config,
        name)

    plt = create_confusion_matrix_plt(confusion_matrix_absolute, f"{name}-absolute-{config['run_name']}", False, dictionary)
    if wandb_logger is not None:
        wandb_logger.log_image(key=f"{name}-absolute", images=[plt])
    plt.close

    confusion_matrix_relative = get_relative_confusion_matrix(confusion_matrix_absolute)
    plt = create_confusion_matrix_plt(confusion_matrix_relative, f"{name}-relative-{config['run_name']}", True, dictionary)
    if wandb_logger is not None:
        wandb_logger.log_image(key=f"{name}-relative", images=[plt])
    plt.close
    
    accuracy = np.trace(confusion_matrix_absolute) * 100 / np.sum(confusion_matrix_absolute)
    if config["dataset_name"] == "Multimodal":
        action_accuracy = np.trace(confusion_matrix_absolute[:4, :4]) * 100 / np.sum(confusion_matrix_absolute[:4, :4])
        color_accuracy = np.trace(confusion_matrix_absolute[13:19, 13:19]) * 100 / np.sum(confusion_matrix_absolute[13:19, 13:19])
        object_accuracy = np.trace(confusion_matrix_absolute[4:13, 4:13]) * 100 / np.sum(confusion_matrix_absolute[4:13, 4:13])
    else:
        action_accuracy  = np.trace(confusion_matrix_absolute[1:6, 1:6]) * 100 / np.sum(confusion_matrix_absolute[1:6, 1:6])
        material_accuracy = np.trace(confusion_matrix_absolute[6:8, 6:8]) * 100 / np.sum(confusion_matrix_absolute[6:8, 6:8])
        color_accuracy   = np.trace(confusion_matrix_absolute[8:17, 8:17]) * 100 / np.sum(confusion_matrix_absolute[8:17, 8:17])
        object_accuracy   = np.trace(confusion_matrix_absolute[17:22, 17:22]) * 100 / np.sum(confusion_matrix_absolute[17:22, 17:22])
        if wandb_logger is not None:
            wandb_logger.log_metrics({f"{name}_material_accuracy": material_accuracy})
    if wandb_logger is not None:
        wandb_logger.log_metrics({f"{name}_sentence_wise_accuracy": sentence_wise_accuracy,
                f"{name}_accuracy": accuracy,
                f"{name}_action_accuracy": action_accuracy,
                f"{name}_color_accuracy": color_accuracy,
                f"{name}_object_accuracy": object_accuracy})
    return sentence_wise_accuracy


def test_supervised(config, wandb_logger, model, datamodule):
    """Test the supervised model.

    Args:
        config (dict): Configuration dictionary.
        wandb_logger (WandbLogger): WandB logger.
        model (LstmClassifier): Trained supervised model.
        datamodule (DataModule): DataModule.
    """
    evaluate(config, wandb_logger, model, datamodule.train_dataloader(), "Final-training")
    evaluate(config, wandb_logger, model, datamodule.val_dataloader()[0], "Final-validation")

    i = config["visible_objects"]
    if config["dataset_name"] == "Multimodal":
        test_data = datamodule.create_dataset(i, 0, 0, False, "constant-test")
        test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False,
                                    num_workers=config["num_workers"])
        sentence_wise_accuracy = evaluate(config, wandb_logger, model, test_loader, f"V{i}-test")
        print_with_time(f"Test accuracy: {np.mean(sentence_wise_accuracy):8.4f}%")

        gen_test_data = datamodule.create_dataset(i, 0, 0, False, "generalization-test")
        gen_test_loader = DataLoader(dataset=gen_test_data, batch_size=config["batch_size"], shuffle=False,
                                        num_workers=config["num_workers"])
    else:
        # no constant test
        gen_test_loader = datamodule.test_dataloader()
    sentence_wise_accuracy_gen = evaluate(config, wandb_logger, model, gen_test_loader, f"V{i}-generalization-test")
    print_with_time(f"Generalization test accuracy: {np.mean(sentence_wise_accuracy_gen):8.4f}%")


def main(args):
    """Main function.

    Args:
        args (argparse.Namespace): Arguments.
    """
    pl.seed_everything(42, workers=True) # for reproducibility
    config = load_config(args.config, args.debug)
    config["debug"] = args.debug # for the wandb logger
    config["mode"] = args.mode 
    print("Config:", config)

    if not args.no_log:
        wandb_logger = WandbLogger(
            project=config["wandb_project"],
            name=config["run_name"],
            save_dir=config["output_dir"],
            anonymous="allow",
        )
        wandb_logger.experiment.config.update(config)
    else:
        wandb_logger = None # no logging

    if args.mode == "supervised":
        if args.unsupervised_model is None:
            supervised_model, supervised_data = train_supervised(config, wandb_logger)

        else:
            unsupervised_model = load_model(args.unsupervised_model, is_unsupervised=True)
            supervised_model, supervised_data = train_supervised(config, wandb_logger, unsupervised_model.encoder)

    else:
        # First Train Unsupervised model
        unsupervised_model = train_unsupervised(config, wandb_logger)
        if args.mode == "unsupervised":
            # Stop here if only unsupervised training is required
            print_with_time("Unsupervised training finished.")
            return
        supervised_model, supervised_data = train_supervised(config, wandb_logger, unsupervised_model.encoder)

    test_supervised(config, wandb_logger, supervised_model, supervised_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
