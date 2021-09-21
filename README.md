# Pretraining Wide Residual Network
This repository contains code for pretraining Wide Residual Network (WRN) [1] on downsampled [2] ImageNet 32x32, 
ImageNet 64x64, and ImageNet 224x224 using cross-entropy and triplet loss [3].

## Environment setup
For creating conda environment, a yml  file `tf2.yml` is provided for replicating setup.
```bash
conda env create -f tf2.yml
conda activate tf2
```

## Data preparation
ImageNet full dataset can be downloaded from [link](http://image-net.org/download-images). After downloading, set the 
path of base_dir in `data.py`.

ImageNet 32x32  and ImageNet 64x64 datasets can be generated either using scripts provided by 
 [Downsampled ImageNet](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts) or 
[TensorFlow  datasets](https://www.tensorflow.org/datasets/catalog/imagenet_resized) package. The tensorflow_datasets 
package can be installed using pip:
```bash
pip install tensorflow_datasets
```
The current version of `tensorflow_datasets=4.4.0` package has a broken link for downloading  ImageNet 32x32  and ImageNet 
64x64. The workaround is available at [GitHub](https://github.com/tensorflow/datasets/issues/3257). 

## Pretraining
For pretraining from scratch using different setups, `pretrain.py` can be used. Details of self-explanatory commandline 
arguments can be seen by passing `--help` to it.
```bash
 python pretrain.py --help
 
       USAGE: pretrain.py [flags]
flags:

pretrain.py:
  --bs: batch_size
    (default: '128')
    (an integer)
  --d: <imagenet_resized/32x32|imagenet_resized/64x64|imagenet-full>: dataset
    (default: 'imagenet_resized/32x32')
  --e: number of epochs
    (default: '50')
    (an integer)
  --g: gpu id
    (default: '0')
  --lbl: <lda|knn>: Specify labelling method either LDA or KNN.
    (default: 'lda')
  --lr: learning_rate
    (default: '0.001')
    (a number)
  --lt: <cross-entropy|triplet>: loss_type  either cross-entropy  or triplet.
    (default: 'cross-entropy')
  --margin: margin for triplet loss
    (default: '1.0')
    (a number)
  --n: network
    (default: 'wrn-28-2')
  --[no]sw: save weights
    (default: 'false')

Try --helpfull to get a list of all flags.
 ```

Pretrained weights will be saved into `weights/` directory. We also provide pretrained weights. They can be downloaded 
from [releases](https://github.com/attaullah/Pretraining-WideResNet/releases) and saved into `weights/` directory. Path of downloaded weights can be set in `wrn.py`.

## Example usage
For using pretrained weights, an example notebook is provided . For more details, see
[cifar_example.ipynb](cifar_example.ipynb).

## Citation 
If you use the provided weights, kindly cite our paper.
```
@misc{sahito2021better,
      title={Better Self-training for Image Classification through Self-supervision}, 
      author={Attaullah Sahito and Eibe Frank and Bernhard Pfahringer},
      year={2021},
      eprint={2109.00778},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## References
1. Wide Residual Networks. Sergey Zagoruyko and Nikos Komodakis. In British
Machine Vision Conference 2016. British Machine Vision Association, 2016.
2. A downsampled variant  of ImageNet as an alternative to the CIFAR datasets. Patryk Chrabaszcz, Ilya Loshchilov, 
and Frank Hutter.  [arXiv preprint arXiv:1707.08819](https://arxiv.org/abs/1707.08819), 2017 .
3. Distance metric learning for large margin nearest neighbour classification. Kilian Q Weinberger and Lawrence K Saul.
Journal of Machine Learning Research,  10(2), 2009.


## License
[MIT](https://github.com/attaullah/Pretraining-WideResNet/blob/main/LICENSE)
