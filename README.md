
# Improving Deep Learning with Differential Privacy using Gradient Encoding and Denoising

This repository is the official implementation of [Improving Deep Learning with Differential Privacy using Gradient Encoding and Denoisin]


## Requirements

To install requirements:

```setup
conda env create -f env.yml
cd torchsearchsorted/
pip install .
```
Download https://drive.google.com/file/d/1k04lW5_IGMnHxfJMd-KR7Md9KlOgKtqi/view?usp=sharing to the main directory. 

Please refer to https://github.com/aliutkus/torchsearchsorted if you had problem installing torchsearchsorted.


## Training

To train the model(s) in the paper, run this command:

```train and test
 python MNIST.py --noisemodel studentt --noiseparams 9 0 1.0 --epochs 100 --microbatch 1 --batch 256 --ngpus 4 --nprocs 50 --clip 1.0 --seed 0  --quantization  --quantclip 1.0 --errcrt --quantmultiplier 4 --distancemultiplier 500 --distancethresh  0.700000 #MNIST
python CIFAR10.py --noisemodel studentt --noiseparams 9 0 0.9 --epochs 300 --microbatch 2 --batch 256 --ngpus 4 --nprocs 40 --clip 1.0 --seed 0  --quantization  --quantclip 1.0 --errcrt --quantmultiplier 4 --distancemultiplier 500 --distancethresh  0.700000 #CIFAR
 ```

> noise model will define the probability distributions used for privatizing the gradient vector, noiseparams is the parameters for the noise model.

## Results

Our model achieves the following performance on :

### [CIFAR]

| Model name         | Accuracy       | eps     |
| ------------------ |----------------|---------|
| This work          |     55%        | 3.6     |
| DPSGD              |     55%        | 5       |

### [MNIST]

| Model name         | Accuracy       | eps     |
| ------------------ |----------------|---------|
| This work          |     96%        | 2.5     |
| DPSGD              |     96%        | 5       |
