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

# Model Specifications
The base model is a 256x512 neural network. It learns very fast but struggles with complicated text, and excels at reproducing very noticable patterns in text.
It uses the GPT-4 Tokenizer as it is very fast, even on low end devices. There isn't much else to say.
