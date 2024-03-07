# CA-Jaacard

Official PyTorch implementation of [CA-Jaccard: Camera-aware Jaccard Distance for Person Re-identification](https://arxiv.org/abs/2311.10605) (CVPR 2024).

## Updates
- 2024/03 The code is released.
- 2024/02/27 CA-Jaccard distance is accepted by CVPR2024! 🎉🎉
- 2023/11/17 CA-Jaccard distance publicly available on arxiv.

## Overview

![overview](Overview.jpg)
>In this paper, we propose a novel camera-aware Jaccard (CA-Jaccard) distance that leverages camera information to enhance the reliability of Jaccard distance. 
Specifically, we introduce camera-aware k-reciprocal nearest neighbors (CKRNNs) to find k-reciprocal nearest neighbors on the intra-camera and inter-camera ranking lists, which improves the reliability of relevant neighbors and guarantees the contribution of inter-camera samples in the overlap. Moreover, we propose a camera-aware local query expansion (CLQE) to exploit camera variation as a strong constraint to mine reliable samples in relevant neighbors and assign these samples higher weights in overlap to further improve the reliability. Our CA-Jaccard distance is simple yet effective and can serve as a general distance metric for person re-ID methods with high reliability and low computational cost. Extensive experiments demonstrate the effectiveness of our method.

## Getting Started

### Installation

```shell
git clone https://github.com/chen960/CA-Jaccard
cd CA-Jaccard
pip install -r requirement.txt
```

### Preparing Datasets

```shell
mkdir data
```

Download the datasets Market-1501, MSMT17, and VeRi-776 to `CA-Jaccard/data`.
The directory should look like:

```
CA-Jaccard/data
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
└── veri
    └── VeRi776
```

## Training (Clustering scene)

We utilize 4 RTX 3090 GPUs for training.

We use 256x128 sized images for Market-1501 and MSMT17 and 224x224 sized images for VeRi-776.

Sometimes setting both k2-intra and k2-inter to 3 can achieve better performance.

```bash
# Market1501
CUDA_VISIBLE_DEVICES=1,2,3,4 python train_caj.py -d market1501  -eps 0.4  --iters 200  --ckrnns --clqe --k2-intra 2 --k2-inter 4 --logs-dir logs/CC+CAJ_market1501  
# msmt17
CUDA_VISIBLE_DEVICES=1,2,3,4 python train_caj.py -d msmt17 --eps 0.6  --iters 400 --ckrnns --clqe --k2-intra 2 --k2-inter 4 --logs-dir logs/CC+CAJ_msmt17
# veri
CUDA_VISIBLE_DEVICES=1,2,3,4 python train_caj.py -d veri --eps 0.6  --iters 400 --height 224 --width 224 --ckrnns --clqe --k2-intra 2 --k2-inter 4 --logs-dir logs/CC+CAJ_veri
```

## Testing

We use a single RTX 3090 GPU for testing.

You can download pre-trained weights from this [link]().


```bash
# market1501
CUDA_VISIBLE_DEVICES=1 python test.py -d market1501 --resume ./pretrained_models/CC+CAJ_market1501_84.8.tar
# msmt17
CUDA_VISIBLE_DEVICES=1 python test.py -d msmt17 --resume ./pretrained_models/CC+CAJ_msmt17_42.8.tar
# veri
CUDA_VISIBLE_DEVICES=1 python test.py -d veri --resume ./pretrained_models/CC+CAJ_veri_43.1.tar
```

## Re-ranking
We use a single RTX 3090 GPU for re-ranking.

Note that reordering on the MSMT17 and VeRi-776 datasets requires at least 400GB memory.

You can download baseline pretrained weights from this [link]().

```bash
# market1501
CUDA_VISIBLE_DEVICES=1 python test.py -d market1501 --resume ./pretrained_models/CC+CAJ_market1501_84.8.tar --rerank --ckrnns --clqe
# msmt17
CUDA_VISIBLE_DEVICES=1 python test.py -d msmt17 --resume ./pretrained_models/CC+CAJ_msmt17_42.8.tar --rerank --ckrnns --clqe
# veri
CUDA_VISIBLE_DEVICES=1 python test.py -d veri --resume ./pretrained_models/CC+CAJ_veri_43.1.tar --rerank --ckrnns --clqe
```

## Acknowledgement

Some parts of the code is borrowed from [Cluster-Contrast](https://github.com/alibaba/cluster-contrast-reid).

## Citation

If you find this code useful for your research, please consider citing our paper:

````BibTex
@inproceedings{yiyu2024caj,
  title={CA-Jaccard: Camera-aware Jaccard Distance for Person Re-identification},
  author={Chen, Yiyu and Fan, Zheyi and Chen, Zhaoru and Zhu, Yixuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
````
