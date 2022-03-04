# XMAM

```
python parameterBoard.py --num_nets 200 --part_nets_per_round 30 --fl_round 100 --dataname cifar10 --model vgg9 --load_premodel True --partition_strategy hetero-dir --dir_parameter 0.5 --malicious_ratio 0.2 --backdoor_type edge-case --poisoned_portion 0.3 --attack_mode pgd --defense_method none --device cuda:0
```

We can change the parameter `--backdoor_type` (trigger|semantic|edge-case) and `--defense_method` (none|krum|multi-krum|ndc|rfa|rsa|XMAM) to reproduce the Fig.9 in our paper:
![image](https://user-images.githubusercontent.com/88427588/156745935-06178c8e-ca51-4cd2-8ac0-72d40477ac35.png)
