import copy
import torch
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

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


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        #data_dir = '../data/cifar/'
        data_dir = root='./CIFAR10'
        transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                  ]) # media an std del CIFA10
        transform_test = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                  ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transform_test)
        
        user_groups = cifar_iid(train_dataset, args.num_users)
        
    return train_dataset, test_dataset, user_groups



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
  direc ="/content/drive/MyDrive/FL2022/ULTIMATE/Results/"
  plt.savefig(f"{direc}+{tlt}.png")
  plt.show()
  
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp