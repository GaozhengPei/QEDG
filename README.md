# This repository contains the code release for **Exploring Query Efficient Data Generation towards Data-free Model Stealing in Hard Label Setting**
![Loss](./image/var&similarity.png)
# How to run
## Obtain a target model 
Train the substitute model first. We take the MNIST dataset as an example.The weight of the target model will be saved in ```./target_model_weight```
```
python train_scratch.py --dataset=mnist --epochs=50 --net=resnet34
```
## Model stealing
Use an substitute model to steal the target model. The weight of the target model will be saved in ```./substitute_model_weight```. The data synthesized by the generator is stored under path ```images_generated```
```
sh scripts/mnist.sh
```
```
sh scripts/fmnist.sh
```
```
sh scripts/svhn.sh
```
```
sh scripts/cifar10.sh
```
# Citation
```
@inproceedings{pei2025exploring,
  title={Exploring query efficient data generation towards data-free model stealing in hard label setting},
  author={Pei, Gaozheng and Lyu, Shaojie and Ma, Ke and Yang, Pinci and Xu, Qianqian and Sun, Yingfei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={667--675},
  year={2025}
}
```
