# Unsupervised-Learning-for-Generalization
Repo of Master thesis Unsupervised Learning for Out-of-Distribution Generalization on a Multimodal Sequence Dataset

# Usage
## Prerequisites
- Python 3.8
- PyTorch 2.0.0
- PyTorch Lightning 1.7.7
- WandB 0.13.2
- OpenCV 4.7.0
- Numpy 1.23.2

See ``requirements.txt`` for complete list of dependencies.
I wouldn't recommend using the ``requirements.txt`` file directly, as it might have some redundant dependencies. Instead, install the dependencies manually.
**TODO: create correct requirements.txt**

Note: To use the ``wandb`` module, you need to have a WandB account and have the ``WANDB_API_KEY`` environment variable set to your API key. See https://docs.wandb.ai/quickstart for more information. Also make sure to set the ``project`` and ``username`` variable in the config file to your project name.

## Training
``python3 train.py --config <config file> --mode <mode> [--unsupervised-model <model>] [--debug <0/1>] [--no-log <0/1>]``  
- ``config``: path to config file (see ``configs`` folder)
- ``mode``: one off ``supervised``, ``unsupervised`` or ``both``.
    - ``supervised``: only supervised training with masking loss if ``unsupervised-model`` is given use that as pretrained model otherwise run without pretraining
    - ``unsupervised``: only unsupervised training with next frame prediction
    - ``both``: first unsupervised pretraining, then supervised training
- ``unsupervised-model``: path to pretrained model for supervised training
- ``no-log``: if set to 1 or True, will not log to WandB
- ``debug``: if set to 1 or True, will run in debug mode (only 1 epoch, few samples)

# Credits
- Starting point of implementation was Volquardsen et al's https://github.com/Casparvolquardsen/Compositional-Generalization-in-Multimodal-Language-Learning 
- ConvLSTM implementation strongely based on https://github.com/holmdk/Video-Prediction-using-PyTorch.git