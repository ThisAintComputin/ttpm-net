# ttpm-net
A neural network that is able to learn patterns in text and reproduce them with as low as 7 seconds of training on low end GPUs.

# Quick Start
To start using TTPM, you can start by cloning this repository and installing Python on your computer or server. After Python is installed, please install these dependencies with PIP:
```
pytorch
tiktoken
numpy
```
After you have installed all of the dependencies, you can put your training data in training.txt, run main.py, and begin training.

# Pretrained Files
In case you don't want to train a brand new model, we have already packaged a pre-trained save file.
To use it, simply delete any existing model.pth file, and rename wikipedia20k.pth to model.pth.
This save file is trained on 10k epochs.
