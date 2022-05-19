# XMAM:X-raying Models with A Matrix to Reveal Backdoor Attacks for Federated Learning

Federated Learning can better protect data privacy because the parameter server only collects the client model and does not touch the local data of the client. However, its basic aggregation algorithm FedAvg is vulnerable to Byzantine client attacks. In response to this problem, many studies have proposed different aggregation algorithms, but these aggregation algorithms have insufficient defensive capabilities, and the model assumptions do not fit the reality. Therefore, we propose a new type of Byzantine robust aggregation algorithm. Different from the existing aggregation algorithms, our method focuses on detecting the probability distribution of the Softmax layer. Specifically, after collecting the client model, the parameter server obtains the Softmax layer probability distribution of the model through the generated matrix to map the updated part of the model, and eliminates the client model with abnormal distribution. The experimental results show that without reducing the accuracy of FedAvg, the Byzantine tolerance rate is increased from 40% to 45% in convergence prevention attacks, and the defense against edge-case backdoor attacks is realized for the first time in backdoor attacks. In addition, according to the current state-of-the-art adaptive attack framework, an adaptive attack is designed specifically for our method, and experimental evaluations have been carried out. The experimental results show that our aggregation algorithm can defend at least 30% of Byzantine clients.

Experimental environment:
```
Python                  3.6.13
torch                   1.8.0+cu111
torchvision             0.9.0+cu111
hdbscan                 0.8.27
```

We can run 2000 iterations to get the converged global model （e.g., pre-model）:
```
python parameterBoard.py --lr 0.02 --num_nets 200 --part_nets_per_round 30 --fl_round 2000 --malicious_ratio 0 --dataname cifar10 --model vgg9 --save_model True --device cuda:0
```

or contact me to get the pre-model *cifar10_vgg9.pt*.


To implement Backdoor attacks and test the robustness of different aggregation methods, we can change the parameter `--backdoor_type` (none|trigger|semantic|edge-case), `--attack_mode` (none|pgd|stealthy), and `--defense_method` (none|krum|multi-krum|ndc|rfa|rsa|xmam) to reproduce the Fig.9 in our paper:

```
python parameterBoard.py --lr 0.00036 --num_nets 200 --part_nets_per_round 30 --fl_round 100 --dataname cifar10 --model vgg9 --load_premodel True --partition_strategy hetero-dir --dir_parameter 0.5 --malicious_ratio 0.2 --backdoor_type edge-case --attack_mode pgd --defense_method none --device cuda:0
```

<div align=center>
<img src="https://user-images.githubusercontent.com/88427588/156745935-06178c8e-ca51-4cd2-8ac0-72d40477ac35.png" width="70%"/>
</div>



MIT license

Programmer: Fangjiao Zhang, Qichao Jin

Email: fungiiizhang@163.com zjy@besti.edu.cn

Under review

北京电子科技学院CSP实验室
