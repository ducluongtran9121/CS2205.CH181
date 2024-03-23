from threading import local
import numpy
from .models import AE
from .utils import *
from .CKA import CKA
from sklearn.cluster import KMeans

class Defense_AE():
    def __init__(self) -> None:
        self.model = AE()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = 20
        self.cluster_Kmeans = KMeans(n_clusters=2, random_state=0)
    
    def train(self, local_models):
        message = "[*] Pretrain anti-poisoning model - AE ..."
        print(message); logging.info(message)
        del message
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for data in local_models:
                data = torch.from_numpy(data)
                output = self.model(data)
                loss = self.criterion(output, data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.data
            message = '\tepoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epochs, total_loss/len(local_models))
            print(message); logging.info(message)
            del message
        torch.save(self.model.state_dict(), f'/home/haochu/Documents/Federated-Averaging-PyTorch/defense/AE_e{self.epochs}.pth')
    
    def test_IO(self, local_models, sampled_client_indices):
        message = "[*] Detect poisoning model stage by AE ..."
        print(message); logging.info(message)
        del message

        poisoning_idx = {}
        self.model.eval()
        loss_items = np.array([])

        for it, data in enumerate(local_models):
            data = torch.from_numpy(data)
            output = self.model(data)
            loss = self.criterion(output, data)
            loss_items = np.append(loss_items, [loss.item()], axis=0)
            message = f"\tLocal Model [{it+1}/{len(local_models)}], loss:{loss.item():.4f}"
            print(message); logging.info(message)
            del message
        loss_items = loss_items.reshape(len(local_models), 1)
        kmeans = self.cluster_Kmeans.fit(loss_items)
        y_pred = kmeans.predict(loss_items)
        poisoning_idx = {}
        for idx, is_poisoning in enumerate(y_pred):
            poisoning_idx[sampled_client_indices[idx]] = is_poisoning
        return poisoning_idx

    def test_latentspace(self, global_plr, local_models, sampled_client_indices):
        message = "[*] Detect poisoning model stage by AE ..."
        print(message); logging.info(message)
        del message
        self.model.eval()
        np_cka = CKA()
        local_cka = np.array([])
        global_latentspace = self.model.encode(torch.from_numpy(global_plr)).detach().numpy()
        
        for it, data in enumerate(local_models,1):
            data = torch.from_numpy(data)
            local_latentspace = self.model.encode(data).detach().numpy()
            cka = np_cka.kernel_CKA(local_latentspace, global_latentspace)
            local_cka = np.append(local_cka, [cka], axis=0)
            message = f"[-] {it}: RBF Kernel CKA, between local LS {it} and global LS: {cka}"
            print(message); logging.info(message)
            del message

        local_cka = local_cka.reshape(len(local_models), 1)
        kmeans = self.cluster_Kmeans.fit(local_cka)
        y_pred = kmeans.predict(local_cka)
        poisoning_idx = {}
        for idx, is_poisoning in enumerate(y_pred):
            poisoning_idx[sampled_client_indices[idx]] = is_poisoning
        return poisoning_idx

class FedCC():
    def __init__(self) -> None:
        self.cluster_Kmeans = KMeans(n_clusters=2, random_state=0)
    
    def detect_poisoning(self, global_plr, local_models, sampled_client_indices):
        message = "[*] Detect poisoning model stage by FedCC ..."
        print(message); logging.info(message)
        del message
        np_cka = CKA()
        local_cka = np.array([])
        
        for it, local_plr in enumerate(local_models,1):
            cka = np_cka.kernel_CKA(local_plr, global_plr)
            local_cka = np.append(local_cka, [cka], axis=0)
            message = f"[-] {it}: RBF Kernel CKA, between local PLR {it} and global PLR: {cka}"
            print(message); logging.info(message)
            del message

        local_cka = local_cka.reshape(len(local_models), 1)
        kmeans = self.cluster_Kmeans.fit(local_cka)
        y_pred = kmeans.predict(local_cka)
        poisoning_idx = {}
        for idx, is_poisoning in enumerate(y_pred):
            poisoning_idx[sampled_client_indices[idx]] = is_poisoning
        return poisoning_idx