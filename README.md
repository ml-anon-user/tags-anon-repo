Please note this code is a partial implementation. To run, please install the required dependencies. Thereafter, download and place the data folder in the root directory by using the link provided below. 

The code has been tested on and NVIDIA RTX 3090 GPU with the following settings:

```
Python 3.9
Ubuntu 22.04
CUDA 11.8
PyTorch 2.0.1
PyTorch3D 0.7.4
PyG 2.3.1
```

# Installation requirements

Run the following pip and conda install commands to set up the environment:
```
conda create -n myenv python=3.9
conda activate myenv
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
conda install pyg -c pyg
pip install point-cloud-utils==0.29.6
pip install plyfile
pip install pandas
pip install tensorboard
pip install torchsummary
conda install pytorch-cluster -c pyg
```

# Data

Please download the zipped data folder from the link at [this link](https://1drv.ms/u/s!Ai8vR3oqUKxTgXezyJwU_ywtge_Z?e=CclWa9).

# How to run

## Inference only (no training)
Please run the following command:
```
python test_full.py
```

You should get the results on the terminal. The evaluation code is within ```./utils/valuate.py```. The output from the network is stored at ```./data/results```.

## Train the network
Training the full network is a 3 step process:

First train a single velocitymodule using:
```
python train_dir.py --val_freq=5000 --train_straight_network=False --feat_embedding_dim=256 --decoder_hidden_dim=64
```

Thereafter, using that checkpoint, train the coupled VM stack with the following:
```
python train_straight.py --val_freq=5000 --train_straight_network=True
```

Given the coupled VM stack checkpoint, the full network can be trained using:
```
python train_full.py --val_freq=2000 --train_straight_network=True --feat_embedding_dim=128 --decoder_hidden_dim=
```

The folder for each training run is placed within ```./logs``` and you can access the necessary checkpoints there. 