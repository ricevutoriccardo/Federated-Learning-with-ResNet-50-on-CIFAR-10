import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import random

import GTKutils


torch.manual_seed(0)

g = torch.Generator()
g.manual_seed(0)

np.random.seed(0)


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class GKTClientTrainer(object):
    def __init__(self, 
    model, 
    train_data, 
    test_data, 
    idxs, 
    client_index,
    gpu=1, 
    optimizer="sgd",
    local_batch_size=16, 
    lr=0.01, 
    local_epochs=1, 
    temperature=3.0, 
    alpha=1.0):
    
        self.model = model

        self.gpu = gpu
        self.device = 'cuda' if self.gpu else 'cpu'
        self.optimizer = optimizer
        self.local_batch_size = local_batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.client_index = client_index

        self.trainloader, self.testloader = self.train_test(
            train_data, test_data, list(idxs))  # get train, valid sets

        if self.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9,
                                             nesterov=True, weight_decay=5e-04)
        elif self.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001,
                                              amsgrad=True)

        self.server_logits_dict = dict()

        self.temperature = temperature
        self.alpha = alpha

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = GTKutils.KL_Loss(self.temperature)  

    def update_large_model_logits(self, logits):
        self.server_logits_dict = logits

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def train_test(self, train_data, test_data, idxs):
        
      
        trainloader = DataLoader(DatasetSplit(train_data, idxs),
                                 batch_size=self.local_batch_size, shuffle=True, generator=g,
                                 worker_init_fn=self.seed_worker)
        testloader = DataLoader(test_data,
                                batch_size=self.local_batch_size, shuffle=False, generator=g,
                                worker_init_fn=self.seed_worker)

        return trainloader, testloader
    

    def train(self):
      
        self.model.train()

        extracted_feature_dict = dict()

        logits_dict = dict()

        labels_dict = dict()

        extracted_feature_dict_test = dict()
        labels_dict_test = dict()

        epoch_loss = []

        for epoch in range(1, self.local_epochs + 1):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                log_probs, _ = self.model(images)
                loss_true = self.criterion_CE(log_probs, labels)

                if len(self.server_logits_dict) != 0:
                    large_model_logits = torch.from_numpy(
                      self.server_logits_dict[batch_idx]
                      ).to(self.device)
                    loss_kd = self.criterion_KL(log_probs, large_model_logits)  # knowledge distillation
                    loss = loss_true + self.alpha * loss_kd
                else:
                    loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())

            print('client {} - Update Epoch: {} - Loss: {:.6f}'.format(
                self.client_index, 
                epoch, 
                sum(batch_loss) / len(batch_loss)
                )
              )

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.model.eval()

        for batch_idx, (images, labels) in enumerate(self.trainloader):

            images, labels = images.to(self.device), labels.to(self.device)
            log_probs, extracted_features = self.model(images)

            """
            if dataset too large -> OUT OF MEMORY ERROR
            it is better to run the program on CPU
            """
            extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()

            log_probs = log_probs.cpu().detach().numpy()

            logits_dict[batch_idx] = log_probs
            labels_dict[batch_idx] = labels.cpu().detach().numpy()

        for batch_idx, (images, labels) in enumerate(self.testloader):
            
            test_images, test_labels = images.to(self.device), labels.to(self.device)
            _, extracted_features_test = self.model(test_images)
            extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()

            labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()

        return extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test
