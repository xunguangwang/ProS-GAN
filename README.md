# Prototype-supervised Adversarial Network for Targeted Attack of Deep Hashing
This is the code for our CVPR 2021 paper "Prototype-supervised Adversarial Network for Targeted Attack of Deep Hashing", which formulates a flexible generative architecture for efficient and effective targeted hashing attack. In this repository, we not only provide the implementation of the proposed Prototype-supervised Adversarial Network (i.e., ProS-GAN), but also collect some popular deep hashing methods used in the paper and the previous targeted attack methods in hashing based retrieval.

## Usage
#### Dependencies
- Python 3.7.6
- Pytorch 1.6.0
- Numpy 1.18.5
- Pillow 7.1.2
- CUDA 10.2


#### Train hashing models
Initialize the hyper-parameters in hashing.py following the paper, and then run
```
python hashing.py
```

#### Attack by P2P or DHTA
Initialize the hyper-parameters in dhta.py following the paper, and then run
```
python dhta.py
```

#### Train ProS-GAN
Initialize the hyper-parameters in main.py following the paper, and then run
```
python main.py --train True
```

#### Evaluate ProS-GAN
Initialize the hyper-parameters in main.py following the paper, and then run
```
python main.py --train False --test True
```

## Cite
If you find this work is useful, please cite the following:
```
@inproceedings{wang2021prototype,
	title={Prototype-supervised Adversarial Network for Targeted Attack of Deep Hashing},
	author={Wang, Xunguang and Zhang, Zheng and Wu, Baoyuan and Shen, Fumin and Lu, Guangming},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	year={2021}
}
```
