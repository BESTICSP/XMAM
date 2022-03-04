# import hdbscan
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats.mstats import gmean
from scipy.stats.mstats import hmean
import torchvision
# data, label = make_blobs(n_features=10, n_samples=100, centers=5, random_state=3)
#
# print(data)
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
# cluster_labels = clusterer.fit_predict(data)
#
# print(cluster_labels)

# mali_total = 0
# for i in range(30):
#     mali = 0
#     aaa = np.random.choice(200, size=10, replace=False)
#     for j in range(len(aaa)):
#         if aaa[j] < 0.4*200:
#             mali += 1
#     print("mali_num", mali)
#     mali_total += mali
#
# print("mali total:", mali_total)

# use_cuda = torch.cuda.is_available()
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# device = torch.device("cuda:1" if use_cuda else "cpu")
#
# a = [i for i in range(1000000)]
# a = torch.tensor(a)
# a = a.to(device)
#
# b = torch.argsort(a, descending=True)
# b = b[0:10000]
# print("======a", a)
# print("======b", b)
#
# for i in range(len(a)):
#     if i in b:
#         pass
#     else:
#         a[i]=0
#
# a = a.detach().cpu()
# plt.bar([i for i in range(len(a))], a)
# # plt.savefig("png/net_{}.png".format(net_index))
# plt.show()



# a = [[0, 0, 27],
#         [3, 4, 6],
#         [7, 6, 3],
#         [3, 6, 8]]
# a = np.array(a)
# print(a)
#
# for i in np.nditer(a, op_flags = ['readwrite']):
#     if i > 3 :
#         i[()] = 1
#     else:
#         i[()] = -1
#
# print(a)

aaa = [1,2,3,4,5,6]
selected_node_indices = np.random.choice(6, size=2, replace=False)

print(selected_node_indices)


