import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import pickle


def load_init_data(dataname, datadir):
    if dataname == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root=datadir, train=True, transform=transform_train, download=True)
        test_data  = datasets.MNIST(root=datadir, train=False, transform=transform_test, download=True)

    elif dataname == 'emnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_data = datasets.EMNIST(root=datadir, split="digits", train=True,
                                     transform=transform_train,
                                     download=True)
        test_data = datasets.EMNIST(root=datadir, split="digits", train=False,
                                    transform=transform_test,
                                    download=True)
        print(train_data.data)

    elif dataname == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        train_data = datasets.CIFAR10(root=datadir, train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR10(root=datadir, train=False, transform=transform_test, download=True)

        train_data.targets = np.array(train_data.targets)
        test_data.targets = np.array(test_data.targets)



    elif dataname == 'cifar100':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        train_data = datasets.CIFAR100(root=datadir, train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR100(root=datadir, train=False, transform=transform_test, download=True)
        train_data.targets = np.array(train_data.targets)
        test_data.targets = np.array(test_data.targets)
    return train_data, test_data

def partition_data(dataname, datadir, partition, n_nets, alpha):

    net_dataidx_map = {}

    train_data, test_data = load_init_data(dataname, datadir=datadir)
    train_data_targets = np.array(train_data.targets)
    n_train = train_data.data.shape[0]
    idxs = np.random.permutation(n_train)

    if partition == "homo":
        batch_idxs = np.array_split(idxs, n_nets)
        for i in range(n_nets):
            net_dataidx_map[i] = batch_idxs[i]

    elif partition == "fortest":
        class_we_use = [3,5,4,0,6,7,8,2,1,9]
        data_we_collect = []
        data_list = [[] for i in range(10)]
        for i in range(n_train):
            if train_data.targets[i] in class_we_use:
                for j in class_we_use:
                    if len(data_list[j]) < 500/len(class_we_use) :
                        data_list[j].append(i)
        for i in range(10):
            data_we_collect.extend(data_list[i])
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map[0] = data_we_collect
        for i in range(1,n_nets):
            net_dataidx_map[i] = batch_idxs[i]

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = len(idxs)
        train_data_targets_numpy = train_data_targets[idxs]

        while (min_size < 10) or (dataname == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_data_targets_numpy == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def poisoning_dataset(dataname, data_for_poison, trigger_label, poison_idx):
    remain = []
    if len(poison_idx):
        if dataname == 'mnist':
            width, height = data_for_poison.data.shape[1:]
            channels = 1
            for idx in poison_idx:
                data_for_poison.targets[idx] = trigger_label
                for c in range(channels):
                    data_for_poison.data[idx, width - 3, height - 3] = 255
                    data_for_poison.data[idx, width - 3, height - 2] = 255
                    data_for_poison.data[idx, width - 2, height - 3] = 255
                    data_for_poison.data[idx, width - 2, height - 2] = 255

                remain.append(data_for_poison[idx])

        elif dataname in ('cifar10','cifar100'):
            width, height, channels = data_for_poison.data.shape[1:]
            for idx in poison_idx:
                data_for_poison.targets[idx] = trigger_label
                for c in range(channels):
                    data_for_poison.data[idx, width - 4, height - 2, c] = 255
                    data_for_poison.data[idx, width - 4, height - 3, c] = 255
                    data_for_poison.data[idx, width - 4, height - 4, c] = 255
                    data_for_poison.data[idx, width - 2, height - 4, c] = 255
                    data_for_poison.data[idx, width - 3, height - 4, c] = 255
                remain.append(data_for_poison[idx])

    return remain

def create_train_data_loader(dataname, train_data, trigger_label, posioned_portion, batch_size, dataidxs, malicious=True):
    if malicious == True:
        perm = np.random.permutation(dataidxs)

        if posioned_portion == 0:
            poison_idx = []
            clean_idx = perm
        elif posioned_portion == 1:
            poison_idx = perm
            clean_idx = []
        else:
            perm = perm.tolist()
            poison_idx = perm[0: int(len(perm) * posioned_portion)]
            clean_idx = perm[int(len(perm) * posioned_portion):]

        poisoning_dataset(dataname, train_data, trigger_label, poison_idx)

        whole_data = copy.deepcopy(train_data)
        whole_data.data = whole_data.data[dataidxs]
        whole_data.targets = whole_data.targets[dataidxs]
        ### separate the data of malicious client for stealthy attack
        clean_part_data = copy.deepcopy(train_data)
        clean_part_data.data = clean_part_data.data[clean_idx]
        clean_part_data.targets = clean_part_data.targets[clean_idx]

        poison_part_data = copy.deepcopy(train_data)
        poison_part_data.data = poison_part_data.data[poison_idx]
        poison_part_data.targets = poison_part_data.targets[poison_idx]

        train_data_loader = DataLoader(dataset=whole_data, batch_size=batch_size, shuffle=True)
        if len(clean_idx):
            clean_part_load = DataLoader(dataset=clean_part_data, batch_size=batch_size, shuffle=True)
        else:
            clean_part_load = []
        if len(poison_idx):
            poison_part_loader = DataLoader(dataset=poison_part_data, batch_size=batch_size, shuffle=True)
        else:
            poison_part_loader = []

        return train_data_loader, clean_part_load, poison_part_loader
    else:
        train_data_client = copy.deepcopy(train_data)
        train_data_client.data = train_data.data[dataidxs]
        train_data_client.targets = train_data.targets[dataidxs]

        train_data_loader = DataLoader(dataset=train_data_client, batch_size=batch_size, shuffle=True)

        return train_data_loader

def create_test_data_loader(dataname, test_data, trigger_label, batch_size):

    test_data_ori = copy.deepcopy(test_data)

    poison_idx = [i for i in range(len(test_data.data))]

    poisoning_dataset(dataname, test_data, trigger_label, poison_idx)

    test_data_tri_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    test_data_ori_loader = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)

    return test_data_ori_loader, test_data_tri_loader

###################################################################################### functions for semantic backdoor
def partition_data_semantic(dataname, datadir, partition, n_nets, alpha):

    net_dataidx_map = {}

    train_data, test_data = load_init_data(dataname, datadir=datadir)
    train_data_targets = np.array(train_data.targets)
    n_train = train_data.data.shape[0]
    idxs = [i for i in range(n_train)]
    idxs = np.array(idxs)

    green_car_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
                       19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500, 9744, 47026,
                       1605, 389]
    green_car_test = [32941, 36005, 40138]

    remaining_idxs_tmp = [i for i in idxs if i not in green_car_train + green_car_test]
    mali_clean_data_idxs = np.random.choice(remaining_idxs_tmp, 64-27, replace=False).tolist()   #  64 is adaptive
    mali_data_idxs = mali_clean_data_idxs + green_car_train
    net_dataidx_map[9999] = idxs[mali_data_idxs]  # net_dataidx_map for malicious client
    net_dataidx_map[99991] = idxs[mali_clean_data_idxs]
    net_dataidx_map[99992] = idxs[green_car_train]

    remaining_idxs = [i for i in idxs if i not in green_car_train + green_car_test]
    remaining_idxs = np.random.permutation(remaining_idxs)

    if partition == "homo":
        batch_idxs = np.array_split(remaining_idxs, n_nets)
        for i in range(n_nets):
            net_dataidx_map[i] = batch_idxs[i]

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = len(remaining_idxs)
        train_data_targets_numpy = train_data_targets[remaining_idxs]

        while (min_size < 10) or (dataname == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_data_targets_numpy == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def create_train_data_loader_semantic(train_data, batch_size, dataidxs, clean_idx, poison_idx):

    for idx in poison_idx:
        train_data.targets[idx] = 2    # green car --> bird

    whole_data = copy.deepcopy(train_data)
    whole_data.data = whole_data.data[dataidxs]
    whole_data.targets = whole_data.targets[dataidxs]
    ### separate the data of malicious client for stealthy attack
    clean_part_data = copy.deepcopy(train_data)
    clean_part_data.data = clean_part_data.data[clean_idx]
    clean_part_data.targets = clean_part_data.targets[clean_idx]

    poison_part_data = copy.deepcopy(train_data)
    poison_part_data.data = poison_part_data.data[poison_idx]
    poison_part_data.targets = poison_part_data.targets[poison_idx]

    train_data_loader = DataLoader(dataset=whole_data, batch_size=batch_size, shuffle=True)
    clean_part_load = DataLoader(dataset=clean_part_data, batch_size=batch_size, shuffle=True)
    poison_part_loader = DataLoader(dataset=poison_part_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, clean_part_load, poison_part_loader

def create_test_data_loader_semantic(test_data, semantic_testset, batch_size):

    test_data_ori_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    test_data_semantic_loader = DataLoader(dataset=semantic_testset, batch_size=batch_size, shuffle=True)

    return test_data_ori_loader, test_data_semantic_loader

###################################################################################### functions for edge-case backdoor
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class CIFAR10_Poisoned(data.Dataset):
    """
    The main motivation for this object is to adopt different transform on the mixed poisoned dataset:
    e.g. there are `M` good examples and `N` poisoned examples in the poisoned dataset.

    """

    def __init__(self, root, clean_indices, poisoned_indices, dataidxs=None, train=True, transform_clean=None,
                 transform_poison=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_clean = transform_clean
        self.transform_poison = transform_poison
        self.target_transform = target_transform
        self.download = download
        self._clean_indices = clean_indices
        self._poisoned_indices = poisoned_indices

        cifar_dataobj = datasets.CIFAR10(self.root, self.train, self.transform_clean, self.target_transform, self.download)

        self.data = cifar_dataobj.data
        self.target = np.array(cifar_dataobj.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # we always assume that the transform function is not None
        if index in self._clean_indices:
            img = self.transform_clean(img)
        elif index in self._poisoned_indices:
            img = self.transform_poison(img)
        else:
            img = self.transform_clean(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

def get_edge_dataloader(datadir, batch_size):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_poison = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        AddGaussianNoise(0., 0.05),
    ])
    # data prep for test set
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=transform_train)

    with open('./backdoorDataset/southwest_images_new_train.pkl', 'rb') as train_f:
        saved_southwest_dataset_train = pickle.load(train_f)

    with open('./backdoorDataset/southwest_images_new_test.pkl', 'rb') as test_f:
        saved_southwest_dataset_test = pickle.load(test_f)

    #
    print("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
    sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],),
                                              dtype=int)  # southwest airplane -> label as truck

    print("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
    sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],),
                                             dtype=int)  # southwest airplane -> label as truck

    # downsample the poisoned dataset ###########################
    num_sampled_poisoned_data_points = 100  # N
    samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                    num_sampled_poisoned_data_points,
                                                    replace=False)
    saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
    sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
    print("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
    ###############################################################

    # downsample the raw cifar10 dataset #################
    num_sampled_data_points = 400  # M
    samped_data_indices = np.random.choice(trainset.data.shape[0], num_sampled_data_points, replace=False)
    tempt_poisoned_trainset = trainset.data[samped_data_indices, :, :, :]
    tempt_poisoned_targets = np.array(trainset.targets)[samped_data_indices]
    print("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
    ########################################################

    ### clean data
    whole_trainset = CIFAR10_Poisoned(root=datadir,
                                         clean_indices=np.arange(tempt_poisoned_trainset.shape[0]),
                                         poisoned_indices=np.arange(tempt_poisoned_trainset.shape[0],
                                                                    tempt_poisoned_trainset.shape[0] +
                                                                    saved_southwest_dataset_train.shape[0]),
                                         train=True, download=True, transform_clean=transform_train,
                                         transform_poison=transform_poison)

    # poisoned_trainset = CIFAR10_truncated(root='./data', dataidxs=None, train=True, transform=transform_train, download=True)
    clean_part_trainset = copy.deepcopy(whole_trainset)
    poison_part_trainset = copy.deepcopy(whole_trainset)

    ####  add poisoned data
    whole_trainset.data = np.append(tempt_poisoned_trainset, saved_southwest_dataset_train, axis=0)
    whole_trainset.target = np.append(tempt_poisoned_targets, sampled_targets_array_train, axis=0)

    poison_part_trainset.data = whole_trainset.data[num_sampled_data_points:]
    poison_part_trainset.target = whole_trainset.target[num_sampled_data_points:]

    clean_part_trainset.data = whole_trainset.data[0:num_sampled_data_points]
    clean_part_trainset.data = whole_trainset.data[0:num_sampled_data_points]

    train_data_loader = torch.utils.data.DataLoader(whole_trainset, batch_size=batch_size, shuffle=True)
    poison_part_loader = torch.utils.data.DataLoader(poison_part_trainset, batch_size=batch_size, shuffle=True)
    clean_part_loader = torch.utils.data.DataLoader(clean_part_trainset, batch_size=batch_size, shuffle=True)

    return train_data_loader, clean_part_loader, poison_part_loader









