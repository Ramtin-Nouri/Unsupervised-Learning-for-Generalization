# Config File
The config file is a simple JSON file that contains all the information needed to run the training.
The config file holds the following information:
- gpus: The number of GPUs to use for training. Or a list of GPUs to use.
- num_workers: The number of workers to use for data loading.
- batch_size: The batch size to use for training.
- epochs: The number of epochs to train for.
- unsupervised_epochs: The number of epochs to unsupervised pretraining for.
- learning_rate: The learning rate to use for training.
- input_length: The number of frames to use as input.
- input_stride: The stride to use for the input frames.
- init_length: The number of frames to use for the initial hidden state.
- use_joints: Whether to use the joint data or not. (Currently not supported anymore)
- predict_ahead: The number of frames to predict ahead for the next frame prediction loss in the unsupervised pretraining.
- output_dir: The directory to save the model checkpoints to.
- data_augmentation: Whether to use data augmentation or not.
- early_stopping_patience: The patience to use for early stopping.

- model: Subsection for the model parameters.
    - convlstm_layers: List of layer sizes for the ConvLSTM.
    - convolution_layers_decoder: List of layer sizes for the convolution layers in the decoder.
    - lstm_num_layers: The number of LSTM layers to use.
    - lstm_hidden_size: The hidden size of the LSTM.
    - dropout_classifier: The dropout rate to use for the classifier.
    - use_resnet: Whether to use a ResNet in the encoder or not.
    - use_mask: Whether to use the masking loss or not.

- dataset: Subsection for the dataset parameters.
    - dataset_name: The name of the dataset to use.
    - data_path: The root directory of the dataset.
    - width: The width of the images.
    - height: The height of the images.
    - num_training_samples: The number of training samples to use. (Not all datasets use this parameter.)
    - num_validation_samples: The number of validation samples to use. (Not all datasets use this parameter.)
    - num_training_samples_unsupervised: The number of training samples to use for unsupervised pretraining. (Not all datasets use this parameter.)
    - num_validation_samples_unsupervised: The number of validation samples to use for unsupervised pretraining. (Not all datasets use this parameter.)
    - visible_objects: The number of visible objects to use for the dataset. (Not all datasets use this parameter.)
    - different_colors: Number of different colors for the dataset. (Not all datasets use this parameter.)
    - different_objects: Number of different objects for the dataset. (Not all datasets use this parameter.)
    - exclusive_colors: Whether to use exclusive colors or not. (Not all datasets use this parameter.)
    - different_actions: Number of different actions for the dataset. (Not all datasets use this parameter.)
    - num_joints: Number of joints for the dataset. (Not all datasets use this parameter.)
    - sentence_length: The length of the sentences for the dataset.
    - multi_sentence: Whether a sentence is composed of multiple sequential actions or not.
    - dictionary_size: The size of the dictionary to use for the dataset.


- wandb: Subsection for the WandB parameters.
    - project: The name of the project to use for WandB.
    - username: The username to use for WandB.