# Implementation of the vanilla federated learning paper:
# Communication-Efficient Learning of Deep Networks from Decentralized Data.
# https://github.com/AshwinRJ/Federated-Learning-PyTorch
import torch
from torchvision import datasets
import torchvision.transforms as transforms

import copy

import numpy as np

torch.manual_seed(2)

g = torch.Generator()
g.manual_seed(2)

np.random.seed(2)


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_average_weights(w, user_groups, idxs_users):
    """
    Returns the weighted average of the weights.
    """
    n_list = []
    for idx in idxs_users:
        n_list.append(len(user_groups[idx]))

    if len(n_list) != len(w):
        print("ERROR IN WEIGHTED AVERAGE!")

    w_avg = copy.deepcopy(w[0])  

    for key in w_avg.keys():
        for i in range(1, len(w)):
            
            w_avg[key] += torch.mul(w[i][key], n_list[i]/sum(n_list)).long()

    return w_avg


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  
    labels = np.array(dataset.targets)


    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_noniid_unbalanced(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset such that
    clients have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard = 1  
    max_shard = 30  

    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def get_dataset(args, unbalanced):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    iid, num_users= args.iid, args.num_users
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # choose the training and test datasets
    train_dataset = datasets.CIFAR10('data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('data', train=False,
                                    download=True, transform=transform_test)

    if iid:
        user_groups = cifar_iid(train_dataset, num_users)
    else:
        if unbalanced:
            user_groups = cifar_noniid_unbalanced(train_dataset, num_users)
        else:
            user_groups = cifar_noniid(train_dataset, num_users)

    return train_dataset, test_dataset, user_groups

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def exp_details(model, optimizer, lr, norm, epochs, iid, frac, local_bs, local_ep, unbalanced, num_users):
    print('\nExperimental details:')
    print(f'    Model     : {model}')
    print(f'    Optimizer : {optimizer}')
    print(f'    Learning  : {lr}')
    print(f'    Normalization  : {norm}')
    print(f'    Global Rounds   : {epochs}\n')

    print('    Federated parameters:')
    if iid:
        print('    IID')
    elif unbalanced:
        print('    Non-IID - unbalanced')
    else:
        print('    Non-IID - balanced')
    print(f'    NUmber of users  : {num_users}')
    print(f'    Fraction of users  : {frac}')
    print(f'    Local Batch size   : {local_bs}')
    print(f'    Local Epochs       : {local_ep}\n')
    return
