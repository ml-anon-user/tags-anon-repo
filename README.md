# Install software requirements
Please run the following pip and conda install commands:
```
conda create -n newenv2023 python=3.7
conda activate newenv2023
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
conda install pyg -c pyg
pip install pytorch-lightning==1.7.6
pip install point-cloud-utils==0.27.0
pip install plyfile
pip install pandas
```

**Please refer to the "How to run" section for either testing a pretrained network or training a new one.**

We have tested our networks with the following setup:
```
Python 3.7
Ubuntu 22.04
CUDA 11.3
PyTorch 1.11.0
PyTorch3D 0.7.1
PyG 2.1.0
```

# Download dataset + pre-trained models

Please download the training and testing datasets and the pretrained model by following this link: [located here](https://1drv.ms/u/s!Ai8vR3oqUKxTd2cwdu5rFGp5F8M?e=fNKvsu) and downloading ```submission.zip```.

Thereafter, unzip ```submission.zip``` and place the contents of the ```/submission``` folder, i.e., the ```/data``` and ```/pretrained``` folders, directly within the root folder where the bash scripts are placed.

# How to run

## Run inference only (without training a new network)
Please make sure ```launcher_test.sh``` has execute permissions and that the ```/data``` and ```/pretrained``` folders are correctly placed in the root directory. Thereafter, run the following command:
```
./launcher_test.sh
```

The evaluation results will be displayed in the terminal. You may further analyse the code within ```Evaluate.py```. The filtered output will be found at ```./data/results```.

## Train the network
Please make sure ```launcher_train.sh``` has execute permissions and that the ```data``` folder is correctly placed in the root directory. Thereafter, run the following command:
```
./launcher_train.sh
```
The folder corresponding to the training run with trained models will appear within ```./logs```. 