import gc
import pickle
import logging

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None
        self.omega_index = {}

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def consolidate(self, Model, Weight, MEAN_pre, epsilon):
        OMEGA_current = {n: p.data.clone().zero_() for n, p in Model.named_parameters()}
        for n, p in Model.named_parameters():
            p_current = p.detach().clone()
            p_change = p_current - MEAN_pre[n]
            # W[n].add_((p.grad**2) * torch.abs(p_change))
            # OMEGA_add = W[n]/ (p_change ** 2 + epsilon)
            # W[n].add_(-p.grad * p_change)
            OMEGA_add = torch.max(Weight[n], Weight[n].clone().zero_()) / (p_change ** 2 + epsilon)
            # OMEGA_add = Weight[n] / (p_change ** 2 + epsilon)
            # OMEGA_current[n] = OMEGA_pre[n] + OMEGA_add
            OMEGA_current[n] = OMEGA_add
        return OMEGA_current

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        # mean_pre = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        # w = {n: p.clone().detach().zero_() for n, p in self.model.named_parameters()}
        self.model.to(self.device)

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
  
                optimizer.zero_grad()
                outputs = self.model(data)
                # print(labels)
                # if -9223372036854775808 in outputs:
                #     print('aaaaaaaa')
                loss = eval(self.criterion)()(outputs, labels)

                loss.backward()
                optimizer.step() 

                if self.device == "cuda": torch.cuda.empty_cache()              
        self.model.to("cpu")
        # omega = self.consolidate(Model=self.model, Weight=w, MEAN_pre=mean_pre, epsilon=0.0001)
        # omega_index = {}
        # for k in omega.keys():
        #     if len(omega[k].view(-1))>1000:
        #         Topk = 100
        #     else:
        #         Topk = int(0.1 * len(omega[k].view(-1)))
        #     Topk_value_index = torch.topk(omega[k].view(-1), Topk)
        #     omega_index[k] = Topk_value_index[1].tolist()
        # self.omega_index = omega_index

    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy
