import torch
from torch import nn
import numpy as np
import pandas as pd

def create_batch_gan(x,batch_size):
    n = len(x)//batch_size
    batch_x = [x[batch_size * i : (i+1) * batch_size].tolist() for i in range(n)]
    batch_x.append(x[batch_size*n:].tolist())
    return batch_x

def preprocessing_gan(dataset):
    columns = dataset.columns[:-1]
    raw_attack = np.array(dataset[dataset["Label"] == 1])[:,:-1]
    raw_benign = np.array(dataset[dataset["Label"] == 0])[:,:-1]
    true_label = dataset["Label"]

    del dataset['Label']
    return dataset, raw_attack, raw_benign, true_label, columns

class Generator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,output_dim),
        )
    
    def forward(self,x):
        x = self.layer(x)
        return x
    
    def generate(self, g_model, input, INPUT_DIM, BATCH_SIZE):
        raw_data, raw_attack, raw_benign, true_label, columns = preprocessing_gan(input)
        batch_attack = create_batch_gan(raw_attack, BATCH_SIZE)
        adv_samples = pd.DataFrame(columns=columns)
        g_model.eval()
        with torch.no_grad():
            for bn in batch_attack:
                attack_b = torch.Tensor(bn)
                if (len(bn) != BATCH_SIZE):
                    BATCH_SIZE = len(bn)
                z = attack_b + torch.Tensor(np.random.uniform(0,1,(BATCH_SIZE,INPUT_DIM)))
                adv_traffic = g_model(z)
                adversarial_attack = torch.clamp(adv_traffic, -0.3, 0.3) + z
                adversarial_attack = torch.clamp(adversarial_attack, 0., 1.)
                adv_data = pd.DataFrame(adversarial_attack, columns=columns)
                adv_samples = pd.concat([adv_samples, adv_data], axis=0)
        adv_samples['Label'] = [0]*adv_samples.shape[0]

        ori_samples = pd.DataFrame(raw_data, columns=columns)
        ori_samples['Label'] = true_label.astype('int32')
        mutated_dataset = pd.concat([ori_samples, adv_samples], axis=0)
        mutated_dataset = mutated_dataset.sample(frac=1).reset_index(drop=True)
        mutated_samples = mutated_dataset.iloc[:,:mutated_dataset.shape[1]-1]
        mutated_labels = mutated_dataset.iloc[:,mutated_dataset.shape[1]-1]

        return mutated_samples, mutated_labels