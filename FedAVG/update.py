import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
  

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu else 'cpu'
        if args.loss == 'NLLLoss':
          self.criterion = nn.NLLLoss()
        if args.loss == 'CrossEntropyLoss':
          self.criterion = nn.CrossEntropyLoss()
        self.trainloader, self.testloader = self.train_test(
          dataset, list(idxs))

    def train_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
      
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.wd)

        for epoch in range(1, self.args.local_ep+1):
            train_loss = 0.0
            correct_train = 0.0
            total=0
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad() 
                log_probs = model(images)  
                loss = self.criterion(log_probs, labels)  
                loss.backward() 
                optimizer.step()  

                train_loss += loss.data.item() 
                _, predicted_outputs = log_probs.max(1)
                total += labels.size(0)
                correct_train += predicted_outputs.eq(labels).sum().item()


            train_loss = train_loss / (batch_idx+1)
            train_acc = correct_train / total

            epoch_loss.append(train_loss)
            print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f} | Accuracy: {:.2f}%'.format(
                    global_round, epoch, train_loss, train_acc*100))
                
      
        print('| Global Round : {} |\tLoss avg: {:.6f} '.format(
                    global_round, sum(epoch_loss) / len(epoch_loss)))
            
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
       
        valid_loss = 0.0
        correct_valid = 0.0 

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images = images.to(self.device) 
            labels = labels.to(self.device)

            # Inference
            outputs = model(images)
            loss = self.criterion(outputs, labels)
            valid_loss += (loss.item() * images.shape[0]) 

            # Prediction
            _, pred_labels = torch.max(outputs.data, 1)
            correct_valid += (pred_labels == labels).float().sum().item()

        valid_loss += valid_loss / len(self.testloader.dataset)
        
        valid_acc = correct_valid / len(self.testloader.dataset)

        return valid_acc, loss
###
def client_update(args, model,client_model, optimizer, train_loader):
    """
    This function updates/trains client model on client data
    """
    epoch=args.local_ep
    device = 'cuda' if args.gpu else 'cpu'
    if args.loss == 'NLLLoss':
      criterion = nn.NLLLoss()
    if args.loss == 'CrossEntropyLoss':
      criterion = nn.CrossEntropyLoss()
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def server_aggregate(global_model, client_models, client_lens, args):
    """
    This function has aggregation method 'mean'
    """
    if(args.iid==0):
      total = sum(client_lens)
    n = len(client_models)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if(args.iid==0):
          global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
        else:
          global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(args, model,global_model, test_dataset):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    device = 'cuda' if args.gpu else 'cpu'
    if args.loss == 'NLLLoss':
      criterion = nn.NLLLoss()
    if args.loss == 'CrossEntropyLoss':
      criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(test_dataset, batch_size=args.local_bs,
                            shuffle=False)  
    with torch.no_grad():
        for data, target in testloader:
          data = data.to(device)
          target = target.to(device)
          output = global_model(data)
          loss = criterion(output, target)

          test_loss += loss.item()
          _, predicted = output.max(1)
          total += target.size(0)
          correct += predicted.eq(target).sum().item()

    test_loss = test_loss/len(testloader.dataset)
    acc = correct/total

    return test_loss, acc

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    test_loss = 0.0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    device = 'cuda' if args.gpu else 'cpu'
    if args.loss == 'NLLLoss':
      criterion = nn.NLLLoss()
    if args.loss == 'CrossEntropyLoss':
      criterion = nn.CrossEntropyLoss()

    testloader = DataLoader(test_dataset, batch_size=args.local_bs,
                            shuffle=False)

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # Inference
        output = model(images)
        loss = criterion(output, labels)
        test_loss += (loss.data.item() * images.shape[0])

        # Prediction
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not args.gpu else np.squeeze(correct_tensor.cpu().numpy())

        for i in range(len(images)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss / len(testloader.dataset)

    accuracy = np.sum(class_correct) / np.sum(class_total)

    return accuracy, test_loss
