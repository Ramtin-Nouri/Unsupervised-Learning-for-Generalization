# Unsupervised-Learning-for-Generalization
Repo of Master thesis "Improving Compositional Generalization
By Learning Concepts Individually".  
Train an action recognition model on Volquardsen et al.'s dataset or on Action Recogntion for Compositional Generalization (ArCoGen), a variation of the CATER dataset.  
Inputs are image sequences and outputs are natural language sentences of the action using a fixed dictionary.

# Abstract
Neural Networks are powerful function approximators, but often fail to generalize beyond their dataset. In this work, we investigate models and techniques for improved out-of-distribution generalization in the context of action recognition tasks. In particular, we focus on compositional generalization, where the data and target labels are compositionally composed of multiple properties.

To improve compositional generalization, we propose two novel approaches, that use a ConvLSTM-based model. The first approach involves unsupervised pre-training on an extended dataset using next frame prediction. We hypothesize, that this method allows the model to learn a more general representation of the data.

Our second approach introduces a new mask, that ignores certain words in the target label. This mask is given to the model as second input, allowing the model to focus on individual properties.

We evaluated the performance of both approaches on Volquardsen et al.'s dataset, and our newly introduced \AG{} dataset. The results of both approaches significantly outperform the baseline on Volquardsen et al.'s dataset on compositional generalization. While the baseline fails completely in the hardest configuration, achieving 0\% compositional generalization accuracy, our masking approach could achieve 74.6\%.
However, the accuracies on our proposed \AG{} dataset were not as strong. While the results of the masking approach were reasonable, giving the more complex dataset, the unsupervised pre-training approach performed poorly and requires further work.
 These findings may pave the way for more effective and generalizable action recognition systems in real-world applications.

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