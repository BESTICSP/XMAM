# XMAM:X-raying Models with A Matrix to Reveal Backdoor Attacks for Federated Learning


We can run 2000 iterations to get the converged global model （e.g., pre-model）:
```
python parameterBoard.py --num_nets 200 --part_nets_per_round 30 --fl_round 2000 --dataname cifar10 --model vgg9 --save_model True --device cuda:0
```

or contact me to get the pre-model *cifar10_vgg9.pt*.


To implement Backdoor attacks and test the robustness of different aggregation methods, we can change the parameter `--backdoor_type` (trigger|semantic|edge-case) and `--defense_method` (none|krum|multi-krum|ndc|rfa|rsa|XMAM) to reproduce the Fig.9 in our paper:

```
python parameterBoard.py --num_nets 200 --part_nets_per_round 30 --fl_round 100 --dataname cifar10 --model vgg9 --load_premodel True --partition_strategy hetero-dir --dir_parameter 0.5 --malicious_ratio 0.2 --backdoor_type edge-case --attack_mode pgd --defense_method none --device cuda:0
```

<div align=center>
<img src="https://user-images.githubusercontent.com/88427588/156745935-06178c8e-ca51-4cd2-8ac0-72d40477ac35.png" width="70%"/>
</div>


Email: Fungiiizhang@163.com (Zhang Fangjiao)
