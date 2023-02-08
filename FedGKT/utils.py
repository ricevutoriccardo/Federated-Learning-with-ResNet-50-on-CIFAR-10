#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from torch.utils.data.dataset import Dataset



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
    

def average_weights(w):
   
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def plotFigure(args, df1):
  os.chdir("/content/drive/MyDrive/FL2022/ULTIMATE/Results")
  if args.norm == "batch_norm":
        n_type="Batch Normalization"
  if args.norm == "group_norm":
        n_type="Group Normalization"

  train_loss = list(df1["train_loss"].values)
  test_loss = list(df1["test_loss"].values)
  train_acc = list(df1["train_acc"].values)
  test_acc = list(df1["test_acc"].values)

  fig, ax = plt.subplots(1, 2, figsize=(20, 7))

  ax[0].plot(range(len(train_loss)), train_loss, linewidth=2, linestyle='-', label="Train")
  ax[0].plot(range(len(test_loss)), test_loss, linewidth=2, linestyle='-', label="Test")

  ax[1].plot(range(len(train_acc)), train_acc, linewidth=2, linestyle='-', label="Train")
  ax[1].plot(range(len(test_acc)), test_acc, linewidth=2, linestyle='-', label="Test")

  ax[0].title.set_text("Average Loss")
  ax[1].title.set_text("Average Accuracy")
  tlt= f"{args.model}_{n_type}_{args.optimizer}_{args.local_bs}_{args.lr}"
  plt.suptitle(tlt, weight="bold")

  ax[0].set(xlabel="# Epochs", ylabel="Loss")
  ax[1].set(xlabel="# Epochs", ylabel="Accuracy")

  ax[0].legend()
  ax[0].grid()
  ax[1].legend()
  ax[1].grid()
  direc ="/content/drive/MyDrive/FL2022/Results/"
  plt.savefig(f"{direc}+{tlt}.png")
  plt.show()


def baseline_data(num):
  
  xtrain, ytrain, xtmp,ytmp = get_cifar10()
  x , y = shuffle_list_data(xtrain, ytrain)

  x, y = x[:num], y[:num]
  transform, _ = get_default_data_transforms(train=True, verbose=False)
  loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)

  return loader

def get_cifar10():
  data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True)
  data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True) 
  
  x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
  x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test

def shuffle_list(data):
  
  for i in range(len(data)):
    tmp_len= len(data[i][0])
    index = [i for i in range(tmp_len)]
    random.shuffle(index)
    data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
  return data

def shuffle_list_data(x, y):
 
  inds = list(range(len(x)))
  random.shuffle(inds)
  return x[inds],y[inds]

class CustomImageDataset(Dataset):
  
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms 

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]
          

def get_default_data_transforms(train=True, verbose=True):
  transforms_train = {
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
  }
  transforms_eval = {    
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  }
  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train['cifar10'].transforms:
      print(' -', transformation)
    print()

  return (transforms_train['cifar10'], transforms_eval['cifar10'])

def baseline_data(num):
 
  xtrain, ytrain, xtmp,ytmp = get_cifar10()
  x , y = shuffle_list_data(xtrain, ytrain)

  x, y = x[:num], y[:num]
  transform, _ = get_default_data_transforms(train=True, verbose=False)
  loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)

  return loader

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("\nData: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
      np.min(labels_test), np.max(labels_test)))

def get_data_loaders(nclients,batch_size,classes_pc=10, real_wd =False ,verbose=True ):

  x_train, y_train, x_test, y_test = get_cifar10()

  if verbose:
    print_image_data_stats(x_train, y_train, x_test, y_test)

  transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

  if real_wd:
    split = split_image_data_realwd(x_train, y_train, n_clients=nclients, verbose = verbose)
  else:  
    split = split_image_data(x_train, y_train, n_clients=nclients, 
          classes_per_client=classes_pc, verbose=verbose)

  split_tmp = shuffle_list(split)

  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
                                                                batch_size=batch_size, shuffle=True) for x, y in split_tmp]

  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

  return client_loaders, test_loader

def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
  
  n_labels = np.max(labels) + 1

  def break_into(n,m):
    to_ret = [1 for i in range(m)]
    for i in range(n-m):
        ind = random.randint(0,m-1)
        to_ret[ind] += 1
    return to_ret

  n_classes = len(set(labels))
  classes = list(range(n_classes))
  np.random.shuffle(classes)
  label_indcs  = [list(np.where(labels==class_)[0]) for class_ in classes]
  
  tmp = [np.random.randint(1,10) for i in range(n_clients)]
  total_partition = sum(tmp)

  class_partition = break_into(total_partition, len(classes))

  class_partition = sorted(class_partition,reverse=True)
  class_partition_split = {}

  for ind, class_ in enumerate(classes):
      class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind],class_partition[ind])]
      

  clients_split = []
  count = 0
  for i in range(n_clients):
    n = tmp[i]
    j = 0
    indcs = []

    while n>0:
        class_ = classes[j]
        if len(class_partition_split[class_])>0:
            indcs.extend(class_partition_split[class_][-1])
            count+=len(class_partition_split[class_][-1])
            class_partition_split[class_].pop()
            n-=1
        j+=1

    classes = sorted(classes,key=lambda x:len(class_partition_split[x]),reverse=True)
    if n>0:
        raise ValueError(" Unable to fulfill the criteria ")
    clients_split.append([data[indcs], labels[indcs]])


  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
      
    if verbose:
      print_split(clients_split)
  
  clients_split = np.array(clients_split)
  
  return clients_split

def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    n_clients : number of clients
    classes_per_client : number of classes per client
    shuffle : True/False => True for shuffling the dataset, False otherwise
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : client data into desired format
  '''
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1


  data_per_client = clients_rand(len(data), n_clients)
  data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]
  
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)
    
  clients_split = []
  c = 0
  for i in range(n_clients):
    client_idcs = []
        
    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
      
    clients_split += [(data[client_idcs], labels[client_idcs])]

  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
      
    if verbose:
      print_split(clients_split)
  
  print_bar_classes(clients_split, n_labels)  
  clients_split = np.array(clients_split)

  return clients_split

def print_bar_classes(clients_split, n_labels):
  di={}
  for i, client in enumerate(clients_split):
    split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    di[f'client{i}']=split
  df= pd.DataFrame.from_dict(di, orient='index',
                       columns=['class1', 'class2','class3','class4','class5','class6', 'class7','class8','class9','class10'])
  #print(df)
  df.plot(kind='bar', stacked=True)
  plt.title("Distribution of Classes into Clients")
  plt.xlabel("Clients")
  plt.xticks([]) 
  plt.legend(loc='upper left')
  plt.ylabel("Number of Images")
  plt.figure(figsize=(40, 40))

def clients_rand(train_len, nclients):
  '''
  train_len: size of the train data
  nclients: number of clients
  
  Returns: to_ret
  
  This function creates a random distribution 
  for the clients, i.e. number of images each client 
  possess.
  '''
  client_tmp=[]
  sum_=0
  for i in range(nclients-1):
    tmp=random.randint(10,100)
    sum_+=tmp
    client_tmp.append(tmp)

  client_tmp= np.array(client_tmp)
  clients_dist= ((client_tmp/sum_)*train_len).astype(int)
  num  = train_len - clients_dist.sum()
  to_ret = list(clients_dist)
  to_ret.append(num)
  return to_ret

def get_data_loaders(nclients,batch_size,classes_pc=10, real_wd =False ,verbose=True ):

  x_train, y_train, x_test, y_test = get_cifar10()

  if verbose:
    print_image_data_stats(x_train, y_train, x_test, y_test)

  transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

  if real_wd:
    split = split_image_data_realwd(x_train, y_train, n_clients=nclients, verbose = verbose)
  else:  
    split = split_image_data(x_train, y_train, n_clients=nclients, 
          classes_per_client=classes_pc, verbose=verbose)

  split_tmp = shuffle_list(split)

  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train), 
                                                                batch_size=batch_size, shuffle=True) for x, y in split_tmp]

  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

  return client_loaders, test_loader

def client_syn(client_model, global_model):
  '''
  This function synchronizes the client model with global model
  '''
  client_model.load_state_dict(global_model.state_dict())


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
                                             replace=False))  # i.i.d. selection from dataset
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


def get_dataset(iid=1, unbalanced=0, num_users=100):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # mean and std of the CIFAR-10 dataset
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # mean and std of the CIFAR-10 dataset
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