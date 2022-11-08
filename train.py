import argparse
from mimetypes import init
from pydoc import ModuleScanner
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
import json
import os
import dataset
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
        input_stride=config_file.get("input_stride", default["input_stride"]),
        use_joints=config_file.get("use_joints", default["use_joints"]),
        output_dir=config_file.get("output_dir", default["output_dir"]),
        learning_rate=config_file.get("learning_rate", default["learning_rate"]),
        convolution_layers_encoder=config_file.get("convolution_layers_encoder", default["convolution_layers_encoder"]),
        convolution_layers_decoder=config_file.get("convolution_layers_decoder", default["convolution_layers_decoder"]),
        dense_layers_encoder=config_file.get("dense_layers_encoder", default["dense_layers_encoder"]),
        dense_layers_decoder=config_file.get("dense_layers_decoder", default["dense_layers_decoder"]),
        lstm_num_layers=config_file.get("lstm_num_layers", default["lstm_num_layers"]),
        lstm_hidden_size=config_file.get("lstm_hidden_size", default["lstm_hidden_size"]),
    )
    # dataset related configs
    dataset = config_file.get("dataset", default["dataset"])
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

    if debug:
        config["num_training_samples"] = 10
        config["num_validation_samples"] = 10
        config["num_training_samples_unsupervised"] = 5
        config["num_validation_samples_unsupervised"] = 5
        config["epochs"] = 1
        config["unsupervised_epochs"] = 1
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
    config["model_path"] = os.path.join(config["output_dir"], run_name)
    os.makedirs(config["model_path"], exist_ok=True)
    
    return config

def train_unsupervised(config, wandb_logger):
    """Train the unsupervised model.

    Args:
        config (dict): Configuration dictionary.
        wandb_logger (WandbLogger): WandB logger.

    Returns:
        LstmAutoencoder: Trained unsupervised model.
    """
    unsupervised_datamodule = dataset.DataModule(config, True)
    unsupervised_model = LstmAutoencoder(config)
    print("Unsupervised model:", unsupervised_model)
    wandb_logger.watch(unsupervised_model, log="all")

    unsupervised_checkpt = pl.callbacks.ModelCheckpoint(
        dirpath=config["model_path"],
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
        filename='unsupervised_{epoch}-{val_loss:.2f}'
    )

    unsupervised_trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["gpus"],
        max_epochs=config["unsupervised_epochs"],
        logger=wandb_logger,
        callbacks=[unsupervised_checkpt],
        log_every_n_steps=1,
        check_val_every_n_epoch=1
    )
    unsupervised_trainer.fit(unsupervised_model, datamodule=unsupervised_datamodule)
    # Log test images
    i = 0
    for batch in unsupervised_datamodule.test_dataloader(): #batch_size = 1
        pred = unsupervised_model.predict(batch)
        if config["use_joints"]:
            pred = pred[0]
        target = batch[0][0][-1]
        wandb_logger.log_image(key=f"test_image", step=i, images=[pred, target], caption=["prediction", "target"])
        i += 1

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
    for key in saved_config:
        if "layers" in key and not "num_layers" in key:
            saved_config[key] = saved_config[key][1:] # Remove the added first layer

    if is_unsupervised:
        model = LstmAutoencoder(saved_config)
    else:
        model = LstmClassifier(saved_config, encoder)
    model.load_state_dict(model_ckpt["state_dict"])
    return model

def train_supervised(config, wandb_logger, encoder):
    """Train the supervised model.

    Args:
        config (dict): Configuration dictionary.
        wandb_logger (WandbLogger): WandB logger.
        encoder (LstmAutoencoder): Trained unsupervised model.

    Returns:
        LstmAutoencoder: Trained supervised model.
        datamodule (DataModule): Supervised datamodule.
    """
    supervised_datamodule = dataset.DataModule(config, False)
    supervised_model = LstmClassifier(config, encoder)
    print("Supervised model:", supervised_model)
    wandb_logger.watch(supervised_model, log="all")

    supervised_checkpt = pl.callbacks.ModelCheckpoint(
        dirpath=config["model_path"],
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        filename='supervised_{epoch}-{val_loss:.2f}'
    )

    supervised_trainer = pl.Trainer(
        accelerator="gpu",
        devices=config["gpus"],
        max_epochs=config["epochs"],
        logger=wandb_logger,
        callbacks=[supervised_checkpt],
        log_every_n_steps=1,
        check_val_every_n_epoch=1
    )
    supervised_trainer.fit(supervised_model, datamodule=supervised_datamodule)

    best = load_model(supervised_checkpt.best_model_path, is_unsupervised=False, encoder=encoder)
    return best,supervised_datamodule


def test_supervised(config, wandb_logger, model, datamodule):
    """Test the supervised model.

    Args:
        config (dict): Configuration dictionary.
        wandb_logger (WandbLogger): WandB logger.
        model (LstmClassifier): Trained supervised model.
        datamodule (DataModule): DataModule.
    """
    # ugly compatibility stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO: avoid passing device, use pytorch lightning
    model.to(device)

    train_confusion_matrix_absolute, final_train_wrong_predictions, final_train_sentence_wise_accuracy = get_evaluation(
        model, datamodule.train_dataloader(),
        device,
        f"Final training")
    val_confusion_matrix_absolute, final_val_wrong_predictions, final_val_sentence_wise_accuracy = get_evaluation(
        model,
        datamodule.val_dataloader(),
        device,
        f"Final validation")

    train_confusion_matrix_relative = get_relative_confusion_matrix(train_confusion_matrix_absolute)
    val_confusion_matrix_relative = get_relative_confusion_matrix(val_confusion_matrix_absolute)

    plt = create_confusion_matrix_plt(train_confusion_matrix_absolute, f"Final-training-absolute-{config['run_name']}", "./logs/", False)
    wandb_logger.log_image(key="Final-training-absolute", images=[plt])
    plt.close()

    plt = create_confusion_matrix_plt(train_confusion_matrix_relative, f"Final-training-relative-{config['run_name']}", "./logs/", True)
    wandb_logger.log_image(key="Final-training-relative", images=[plt])
    plt.close()

    plt = create_confusion_matrix_plt(val_confusion_matrix_absolute, f"Final-validation-absolute-{config['run_name']}", "./logs/", False)
    wandb_logger.log_image(key="Final-validation-absolute", images=[plt])
    plt.close()

    plt = create_confusion_matrix_plt(val_confusion_matrix_relative, f"Final-validation-relative-{config['run_name']}", "./logs/", True)
    wandb_logger.log_image(key="Final-validation-relative", images=[plt])
    plt.close()

    final_train_accuracy = np.trace(train_confusion_matrix_absolute) * 100 / np.sum(train_confusion_matrix_absolute)
    final_train_action_accuracy = np.trace(train_confusion_matrix_absolute[:4, :4]) * 100 / np.sum(
        train_confusion_matrix_absolute[:4, :4])
    final_train_color_accuracy = np.trace(train_confusion_matrix_absolute[13:19, 13:19]) * 100 / np.sum(
        train_confusion_matrix_absolute[13:19, 13:19])
    final_train_object_accuracy = np.trace(train_confusion_matrix_absolute[4:13, 4:13]) * 100 / np.sum(
        train_confusion_matrix_absolute[4:13, 4:13])

    final_val_accuracy = np.trace(val_confusion_matrix_absolute) * 100 / np.sum(val_confusion_matrix_absolute)
    final_val_action_accuracy = np.trace(val_confusion_matrix_absolute[:4, :4]) * 100 / np.sum(
        val_confusion_matrix_absolute[:4, :4])
    final_val_color_accuracy = np.trace(val_confusion_matrix_absolute[13:19, 13:19]) * 100 / np.sum(
        val_confusion_matrix_absolute[13:19, 13:19])
    final_val_object_accuracy = np.trace(val_confusion_matrix_absolute[4:13, 4:13]) * 100 / np.sum(
        val_confusion_matrix_absolute[4:13, 4:13])

    wandb_logger.log_metrics({f"Final_training_sentence_wise_accuracy": final_train_sentence_wise_accuracy,
                f"Final_training_accuracy": final_train_accuracy,
                f"Final_training_action_accuracy": final_train_action_accuracy,
                f"Final_training_color_accuracy": final_train_color_accuracy,
                f"Final_training_object_accuracy": final_train_object_accuracy,

                f"Final_validation_sentence_wise_accuracy": final_val_sentence_wise_accuracy,
                f"Final_validation_accuracy": final_val_accuracy,
                f"Final_validation_action_accuracy": final_val_action_accuracy,
                f"Final_validation_color_accuracy": final_val_color_accuracy,
                f"Final_validation_object_accuracy": final_val_object_accuracy,

                f"Final_training_wrong_predictions": final_train_wrong_predictions,
                f"Final_validation_wrong_predictions": final_val_wrong_predictions})

    cf_matrices_absolute = np.zeros((6, 19, 19))
    cf_matrices_absolute_gen = np.zeros((6, 19, 19))

    #TODO delete duplicate (dataset)
    dataset_mean = [0.7605, 0.7042, 0.6045]
    dataset_std = [0.1832, 0.2083, 0.2902]

    torchvision_mean = [0.485, 0.456, 0.406]
    torchvision_std = [0.229, 0.224, 0.225]

    normal_transform = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    torchvision_transform = transforms.Normalize(mean=torchvision_mean, std=torchvision_std)
    if False: #TODO
        transform = torchvision_transform
    else:
        transform = normal_transform

    num_samples = 10 if config["debug"] else 2000
    for i in range(1, 7):
        test_data = dataset.MultimodalSimulation(path=config["data_path"],
                                            visible_objects=i,
                                            different_actions=config["different_actions"],
                                            different_colors=config["different_colors"],
                                            different_objects=config["different_objects"],
                                            exclusive_colors=config["exclusive_colors"],
                                            part="constant-test",
                                            num_samples=num_samples,
                                            input_length=config["input_length"],
                                            frame_stride=config["input_stride"],
                                            transform=transform,
                                            debug=config["debug"])

        gen_test_data = dataset.MultimodalSimulation(path=config["data_path"],
                                                visible_objects=i,
                                                different_actions=config["different_actions"],
                                                different_colors=config["different_colors"],
                                                different_objects=config["different_objects"],
                                                exclusive_colors=config["exclusive_colors"],
                                                part="generalization-test",
                                                num_samples=num_samples,
                                                input_length=config["input_length"],
                                                frame_stride=config["input_stride"],
                                                transform=transform,
                                                debug=config["debug"])

        # dataloader
        test_loader = DataLoader(dataset=test_data, batch_size=config["batch_size"], shuffle=False,
                                    num_workers=config["num_workers"])
        gen_test_loader = DataLoader(dataset=gen_test_data, batch_size=config["batch_size"], shuffle=False,
                                        num_workers=config["num_workers"])

        confusion_matrix_absolute, wrong_predictions, sentence_wise_accuracy = get_evaluation(model, test_loader, device, f"V{i} test")
        confusion_matrix_absolute_gen, wrong_predictions_gen, sentence_wise_accuracy_gen = get_evaluation(model, gen_test_loader, device, f"V{i} generalization test")

        confusion_matrix_relative = get_relative_confusion_matrix(confusion_matrix_absolute)
        confusion_matrix_relative_gen = get_relative_confusion_matrix(confusion_matrix_absolute_gen)

        plt = create_confusion_matrix_plt(confusion_matrix_absolute,
                                            f"V{i}-test-absolute-{config['run_name']}", "./logs/", False)
        wandb_logger.log_image(key=f"V{i}-test-absolute", images=[plt])
        plt.close()

        plt = create_confusion_matrix_plt(confusion_matrix_relative,
                                            f"V{i}-test-relative-{config['run_name']}", "./logs/", True)
        wandb_logger.log_image(key=f"V{i}-test-relative", images=[plt])
        plt.close()

        plt = create_confusion_matrix_plt(confusion_matrix_absolute_gen,
                                            f"V{i}-generalization-test-absolute-{config['run_name']}", "./logs/", False)
        wandb_logger.log_image(key=f"V{i}-generalization-test-absolute", images=[plt])
        plt.close()

        plt = create_confusion_matrix_plt(confusion_matrix_relative_gen,
                                            f"V{i}-generalization-test-relative-{config['run_name']}", "./logs/", True)
        wandb_logger.log_image(key=f"V{i}-generalization-test-relative", images=[plt])
        plt.close()

        test_accuracy = np.trace(confusion_matrix_absolute) * 100 / np.sum(confusion_matrix_absolute)
        test_action_accuracy = np.trace(confusion_matrix_absolute[:4, :4]) * 100 / np.sum(
            confusion_matrix_absolute[:4, :4])
        test_color_accuracy = np.trace(confusion_matrix_absolute[13:19, 13:19]) * 100 / np.sum(
            confusion_matrix_absolute[13:19, 13:19])
        test_object_accuracy = np.trace(confusion_matrix_absolute[4:13, 4:13]) * 100 / np.sum(
            confusion_matrix_absolute[4:13, 4:13])

        gen_test_accuracy = np.trace(confusion_matrix_absolute_gen) * 100 / np.sum(confusion_matrix_absolute_gen)
        gen_test_action_accuracy = np.trace(confusion_matrix_absolute_gen[:4, :4]) * 100 / np.sum(
            confusion_matrix_absolute_gen[:4, :4])
        gen_test_color_accuracy = np.trace(confusion_matrix_absolute_gen[13:19, 13:19]) * 100 / np.sum(
            confusion_matrix_absolute_gen[13:19, 13:19])
        gen_test_object_accuracy = np.trace(confusion_matrix_absolute_gen[4:13, 4:13]) * 100 / np.sum(
            confusion_matrix_absolute_gen[4:13, 4:13])

        cf_matrices_absolute[i - 1] = confusion_matrix_absolute
        cf_matrices_absolute_gen[i - 1] = confusion_matrix_absolute_gen

        wandb_logger.log_metrics({f"V{i}-test_sentence_wise_accuracy": sentence_wise_accuracy,
                    f"V{i}-test_accuracy": test_accuracy,
                    f"V{i}-test_action_accuracy": test_action_accuracy,
                    f"V{i}-test_color_accuracy": test_color_accuracy,
                    f"V{i}-test_object_accuracy": test_object_accuracy,
                    f"V{i}-generalization_test_sentence_wise_accuracy": sentence_wise_accuracy_gen,
                    f"V{i}-generalization_test_accuracy": gen_test_accuracy,
                    f"V{i}-generalization_test_action_accuracy": gen_test_action_accuracy,
                    f"V{i}-generalization_test_color_accuracy": gen_test_color_accuracy,
                    f"V{i}-generalization_test_object_accuracy": gen_test_object_accuracy,
                    f"V{i}-test_wrong_predictions": wrong_predictions,
                    f"V{i}-generalization_test_wrong_predictions": wrong_predictions_gen})

    test_accuracy = np.sum(np.trace(cf_matrices_absolute, axis1=1, axis2=2)) * 100 / np.sum(cf_matrices_absolute)
    test_action_accuracy = np.sum(np.trace(cf_matrices_absolute[:, :4, :4], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute[:, :4, :4])
    test_color_accuracy = np.sum(np.trace(cf_matrices_absolute[:, 13:19, 13:19], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute[:, 13:19, 13:19])
    test_object_accuracy = np.sum(np.trace(cf_matrices_absolute[:, 4:13, 4:13], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute[:, 4:13, 4:13])

    gen_test_accuracy = np.sum(np.trace(cf_matrices_absolute_gen, axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen)
    gen_test_action_accuracy = np.sum(
        np.trace(cf_matrices_absolute_gen[:, :4, :4], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen[:, :4, :4])
    gen_test_color_accuracy = np.sum(
        np.trace(cf_matrices_absolute_gen[:, 13:19, 13:19], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen[:, 13:19, 13:19])
    gen_test_object_accuracy = np.sum(
        np.trace(cf_matrices_absolute_gen[:, 4:13, 4:13], axis1=1, axis2=2)) * 100 / np.sum(
        cf_matrices_absolute_gen[:, 4:13, 4:13])

    wandb_logger.log_metrics({"test_accuracy": test_accuracy,
                "test_action_accuracy": test_action_accuracy,
                "test_color_accuracy": test_color_accuracy,
                "test_object_accuracy": test_object_accuracy})
    print_with_time(f"Test accuracy: {test_accuracy:8.4f}%")

    wandb_logger.log_metrics({"generalization_test_accuracy": gen_test_accuracy,
                "generalization_test_action_accuracy": gen_test_action_accuracy,
                "generalization_test_color_accuracy": gen_test_color_accuracy,
                "generalization_test_object_accuracy": gen_test_object_accuracy})
    print_with_time(f"Generalization test accuracy: {gen_test_accuracy:8.4f}%")


def main(args):
    pl.seed_everything(42, workers=True) # for reproducibility
    config = load_config(args.config, args.debug)
    print("Config:", config)

    wandb_logger = WandbLogger(
        project=config["wandb_project"],
        name=config["run_name"],
        save_dir=config["output_dir"],
        anonymous="allow",
    )
    wandb_logger.experiment.config.update(config)

    if args.mode == "supervised":
        if args.unsupervised_model is None:
            print_fail("If mode is supervised, you must provide a path to an unsupervised model.")
            exit(1)
        unsupervised_model = load_unsupervised_model(args.unsupervised_model)
    else:
        # First Train Unsupervised model
        unsupervised_model = train_unsupervised(config, wandb_logger)

    if args.mode == "unsupervised":
        print_with_time("Unsupervised training finished.")
        return

    # Use the trained model to get the latent space, dismiss its decoder
    unsupervised_model = unsupervised_model.encoder

    # Train supervised model
    supervised_model, supervised_data = train_supervised(config, wandb_logger, unsupervised_model)

    test_supervised(config, wandb_logger, supervised_model, supervised_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)