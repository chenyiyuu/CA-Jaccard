# CA-Jaacard

Official PyTorch implementation of [CA-Jaccard: Camera-aware Jaccard Distance for Person Re-identification](https://arxiv.org/abs/2311.10605) (CVPR 2024).

## Updates
- 2024/03 The code is released.
- 2024/02/27 CA-Jaccard distance is accepted by CVPR2024! 🎉🎉
- 2024/11/17 CA-Jaccard distance publicly available on arxiv.

## Overview

![overview]()
>In this paper, we propose a novel camera-aware Jaccard (CA-Jaccard) distance that leverages camera information to enhance the reliability of Jaccard distance. 
Specifically, we introduce camera-aware k-reciprocal nearest neighbors (CKRNNs) to find k-reciprocal nearest neighbors on the intra-camera and inter-camera ranking lists, which improves the reliability of relevant neighbors and guarantees the contribution of inter-camera samples in the overlap. Moreover, we propose a camera-aware local query expansion (CLQE) to exploit camera variation as a strong constraint to mine reliable samples in relevant neighbors and assign these samples higher weights in overlap to further improve the reliability. Our CA-Jaccard distance is simple yet effective and can serve as a general distance metric for person re-ID methods with high reliability and low computational cost. Extensive experiments demonstrate the effectiveness of our method.

## Getting Started

### Installation

```shell
git clone https://github.com/yoonkicho/PPLR
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

## Training

We utilize 4 RTX 3090 GPUs for training.
We use 256x128 sized images for Market-1501 and MSMT17 and 224x224 sized images for VeRi-776.

For Market-1501:

```

```

For MSMT17:

```

```

For VeRi-776:

```

```

## Testing 

We use a single RTX 3090 GPU for testing.

You can download pre-trained weights from this [link](https://drive.google.com/drive/folders/1m5wDOJG7qk62PjkoOpTspNmk0nhLc4Vi?usp=sharing).

For Market-1501:

```

```

For MSMT17:

```

```

For VeRi-776:

```

```

## Acknowledgement

Some parts of the code is borrowed from [Cluster Contrast]([https://github.com/yxgeee/SpCL](https://github.com/alibaba/cluster-contrast-reid)).

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
