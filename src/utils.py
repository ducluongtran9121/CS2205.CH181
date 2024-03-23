import os
import logging
from importlib_metadata import sys
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from .GAN import Generator
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

with open('/home/haochu/Documents/Federated-Averaging-PyTorch/config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))

model_name = configs[5]["model_config"]["name"]
num_attackers = configs[2]["fed_config"]["A"]

#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     model.to(gpu_ids[0])
    #     model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        X = self.tensors[0][index] 
        y = self.tensors[1][index]
        return X, y

    def __len__(self):
        return self.tensors[0].size(0)

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

def preprocessing_training_dataset(dataset, dataset_name):
    # remove any inf or nan values
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    # drop irrelevant features 
    if dataset_name in ["CIC-ToN-IoT"]:
        dataset = swap_columns(dataset, 'Label', 'Attack')
        label_encoder = LabelEncoder()
        dataset['Attack'] = label_encoder.fit_transform(dataset['Attack'])
        dataset = dataset.drop(["Flow ID", "Src IP", "Dst IP", "Timestamp", 'CWE Flag Count', 'ECE Flag Cnt'], axis=1)
    column_names = np.array(dataset.columns)
    to_drop = []
    for x in column_names:
        size = dataset.groupby([x]).size()
        #check for cols that only take an unique value
        if (len(size.unique()) == 1):
            to_drop.append(x)
    if dataset_name not in ["Edge"]:
        dataset = dataset.drop(to_drop, axis=1)
    else:
        dataset = dataset.drop(to_drop[:3], axis=1)
    # dataset = dataset.drop(['Unnamed: 0'], axis=1, errors='ignore')
    print("Dataset after processing:", dataset.shape)
    columns = dataset.columns[:-1]
    # X,y of data
    if dataset_name in ["CIC-ToN-IoT"]:
        trainy = dataset['Label']
        trainx = dataset.drop(['Label'], axis=1)
    else:
        trainx = dataset.iloc[:,:dataset.shape[1]-1]
        trainy = dataset.iloc[:,dataset.shape[1]-1]

    # Normalize data
    mms = MinMaxScaler().fit(trainx)
    trainx = mms.transform(trainx)

    if model_name == "CNN":
        trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
    else:
        trainx = np.reshape(trainx, (trainx.shape[0], 1, 5, trainx.shape[1]//5))
    return trainx, trainy.to_numpy(), columns


def create_datasets(dataset_name, num_clients, iid, attack_mode, num_attackers):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    # get dataset 
    if dataset_name in ["CIC-ToN-IoT"]:
        full_dataset = pd.read_csv("/home/haochu/Documents/PoisoningAttack/Dataset/CICToNIoT/Original/CIC-ToN-IoT.csv")
        dataset = full_dataset.sample(frac=0.2)
        training_dataset = dataset.sample(frac=0.75)
        testing_dataset = dataset.drop(training_dataset.index)
    elif dataset_name in ["CICIDS2017"]:
        dataset = pd.read_csv("/home/haochu/Documents/PoisoningAttack/Dataset/CICIDS2017/CICIDS2017_full.csv")
        training_dataset = dataset.sample(frac=0.75)
        testing_dataset = dataset.drop(training_dataset.index)
    elif dataset_name in ["N-BaIoT"]:
        dataset = pd.read_csv("/home/haochu/Documents/PoisoningAttack/Dataset/N-BaIoT/Full/N-baiot5.csv")
        training_dataset = dataset.sample(frac=0.70)
        testing_dataset = dataset.drop(training_dataset.index)
    elif dataset_name in ["Edge"]:
        dataset = pd.read_csv("/home/haochu/Downloads/Edge-IIoTset dataset/Selected/Edge.csv")
        training_dataset = dataset.sample(frac=0.70)
        testing_dataset = dataset.drop(training_dataset.index)
    # Ember dataset
    else:
        path = "/home/haochu/Documents/Federated-Averaging-PyTorch/ember/"
        train_size = 800000
        test_size = 200000
        columns = 2381
        X_train = np.memmap(path+"X_train.dat", dtype=np.float32, mode="r", shape=(train_size, columns))
        y_train = np.memmap(path+"y_train.dat", dtype=np.float32, mode="r", shape=train_size)
        X_test = np.memmap(path+"X_test.dat", dtype=np.float32, mode="r", shape=(test_size, columns))
        y_test = np.memmap(path+"y_test.dat", dtype=np.float32, mode="r", shape=test_size)
        
        train_rows = (y_train != -1)
        X_train = X_train[train_rows]
        y_train = y_train[train_rows]

        X_train, X_removed, y_train, y_removed = train_test_split(X_train, y_train, test_size=0.7, random_state=1)

        test_rows = (y_test != -1)
        X_test = X_test[test_rows]
        y_test = y_test[test_rows]

        X_test, X_removed, y_test, y_removed = train_test_split(X_test, y_test, test_size=0.7, random_state=1)

        print('X_train: ', X_train.shape)
        print('X_test: ', X_test.shape)

        pca = PCA(n_components=50)

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print('X_train_pca:', X_train.shape)
        print('X_test_pca:', X_test.shape)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    

    if dataset_name not in ["Ember"]:
        training_inputs, training_labels, columns = preprocessing_training_dataset(training_dataset, dataset_name)
        testing_inputs, testing_labels, columns = preprocessing_training_dataset(testing_dataset, dataset_name)
    else:
        training_inputs, training_labels = X_train.copy(), y_train.copy()
        testing_inputs, testing_labels = X_test.copy(), y_test.copy()
        
    # split dataset according to iid flag
    if iid:
        # shuffle data
        shuffled_indices = torch.randperm(len(training_inputs))
        training_inputs = training_inputs[shuffled_indices]
        training_labels = torch.Tensor(training_labels)[shuffled_indices]

        # partition data into num_clients
        split_size = len(training_inputs) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )
    

    
    else:
        shuffled_indices = torch.randperm(len(training_inputs))
        training_inputs = training_inputs[shuffled_indices]
        training_labels = torch.Tensor(training_labels)[shuffled_indices]

        # partition data into num_clients
        split_size = len(training_inputs) // (num_clients)
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )

        if dataset_name in ["CIC-ToN-IoT"]:
            num_move = 20000
            buff_move1 = 5000
            buff_move2 = 2500
        else:
            num_move = 3000
            buff_move1 = 1000
            buff_move2 = 500

        
    #         index_0_0 = split_datasets[0][1] == 0
    #         data_0_0 = split_datasets[0][0][index_0_0]
    #         data_0_1 = split_datasets[0][0][~index_0_0]
    #         label_0_0 = split_datasets[0][1][index_0_0]
    #         label_0_1 = split_datasets[0][1][~index_0_0]

    #         # data_0_0_move, data_0_0_base = data_0_0.split([data_0_0.size(0)-1, 1])
    #         # label_0_0_move, label_0_0_base = label_0_0.split([data_0_0.size(0)-1, 1])

    #         data_0_0_move, data_0_0_base = data_0_0.split([num_move + buff_move1, data_0_0.size(0) - buff_move1 - num_move])
    #         label_0_0_move, label_0_0_base = label_0_0.split([num_move + buff_move1, label_0_0.size(0) - buff_move1 - num_move])


    #         index_1_0 = split_datasets[1][1] == 0
    #         data_1_0 = split_datasets[1][0][index_1_0]
    #         data_1_1 = split_datasets[1][0][~index_1_0]
    #         label_1_0 = split_datasets[1][1][index_1_0]
    #         label_1_1 = split_datasets[1][1][~index_1_0]

    #         data_1_1_move, data_1_1_base = data_1_1.split([num_move + buff_move1, data_1_1.size(0) - buff_move1 - num_move])
    #         label_1_1_move, label_1_1_base = label_1_1.split([num_move + buff_move1 , label_1_1.size(0) - buff_move1 - num_move])

    #         # data_1_1_move, data_1_1_base = data_1_1.split([data_1_1.size(0)-1, 1])
    #         # label_1_1_move, label_1_1_base = label_1_1.split([data_1_1.size(0)-1, 1])


    #         data0_new = torch.cat((data_0_0_base, data_0_1, data_1_1_move), 0)
    #         label0_new = torch.cat((label_0_0_base, label_0_1, label_1_1_move), 0)


    #         data1_new = torch.cat((data_1_1_base, data_1_0, data_0_0_move), 0)
    #         label1_new = torch.cat((label_1_1_base, label_1_0, label_0_0_move), 0)


    #         list_split_datasets_0 = list(split_datasets[0])
    #         list_split_datasets_1 = list(split_datasets[1])



    #         list_split_datasets_0[0] = torch.clone(data0_new)
    #         list_split_datasets_0[1] = torch.clone(label0_new)        

    #         list_split_datasets_1[0] = torch.clone(data1_new)
    #         list_split_datasets_1[1] = torch.clone(label1_new)

    #         split_datasets[0] = tuple(list_split_datasets_0)
    #         split_datasets[1] = tuple(list_split_datasets_1)

    # # ------------------------------------------------
    #         index_2_0 = split_datasets[2][1] == 0
    #         data_2_0 = split_datasets[2][0][index_2_0]
    #         data_2_1 = split_datasets[2][0][~index_2_0]
    #         label_2_0 = split_datasets[2][1][index_2_0]
    #         label_2_1 = split_datasets[2][1][~index_2_0]

    #         # data_2_0_move, data_2_0_base = data_2_0.split([data_2_0.size(0)-1, 1])
    #         # label_2_0_move, label_2_0_base = label_2_0.split([data_2_0.size(0)-1, 1])

    #         data_2_0_move, data_2_0_base = data_2_0.split([num_move+ buff_move2, data_2_0.size(0) - buff_move2 - num_move])
    #         label_2_0_move, label_2_0_base = label_2_0.split([num_move + buff_move2, label_2_0.size(0) - buff_move2 - num_move])


    #         index_3_0 = split_datasets[3][1] == 0
    #         data_3_0 = split_datasets[3][0][index_3_0]
    #         data_3_1 = split_datasets[3][0][~index_3_0]
    #         label_3_0 = split_datasets[3][1][index_3_0]
    #         label_3_1 = split_datasets[3][1][~index_3_0]

    #         data_3_1_move, data_3_1_base = data_3_1.split([num_move + buff_move2, data_3_1.size(0) - buff_move2 - num_move])
    #         label_3_1_move, label_3_1_base = label_3_1.split([num_move + buff_move2, label_3_1.size(0) - buff_move2 - num_move])

    #         # data_3_1_move, data_3_1_base = data_3_1.split([data_3_1.size(0)-1, 1])
    #         # label_3_1_move, label_3_1_base = label_3_1.split([data_3_1.size(0)-1, 1])


    #         data2_new = torch.cat((data_2_0_base, data_2_1, data_3_1_move), 0)
    #         label2_new = torch.cat((label_2_0_base, label_2_1, label_3_1_move), 0)


    #         data3_new = torch.cat((data_3_1_base, data_3_0, data_2_0_move), 0)
    #         label3_new = torch.cat((label_3_1_base, label_3_0, label_2_0_move), 0)


    #         list_split_datasets_2 = list(split_datasets[2])
    #         list_split_datasets_3 = list(split_datasets[3])



    #         list_split_datasets_2[0] = torch.clone(data2_new)
    #         list_split_datasets_2[1] = torch.clone(label2_new)        

    #         list_split_datasets_3[0] = torch.clone(data3_new)
    #         list_split_datasets_3[1] = torch.clone(label3_new)

    #         split_datasets[2] = tuple(list_split_datasets_2)
    #         split_datasets[3] = tuple(list_split_datasets_3)

# -----------------------------------------------
        if num_attackers == 2:
            index_4_0 = split_datasets[4][1] == 0
            data_4_0 = split_datasets[4][0][index_4_0]
            data_4_1 = split_datasets[4][0][~index_4_0]
            label_4_0 = split_datasets[4][1][index_4_0]
            label_4_1 = split_datasets[4][1][~index_4_0]

            data_4_0_move, data_4_0_base = data_4_0.split([data_4_0.size(0)-1, 1])
            label_4_0_move, label_4_0_base = label_4_0.split([data_4_0.size(0)-1, 1])

            # data_4_0_move, data_4_0_base = data_4_0.split([num_move, data_4_0.size(0) - num_move])
            # label_4_0_move, label_4_0_base = label_4_0.split([num_move, label_4_0.size(0) - num_move])


            index_5_0 = split_datasets[5][1] == 0
            data_5_0 = split_datasets[5][0][index_5_0]
            data_5_1 = split_datasets[5][0][~index_5_0]
            label_5_0 = split_datasets[5][1][index_5_0]
            label_5_1 = split_datasets[5][1][~index_5_0]

            # data_5_1_move, data_5_1_base = data_5_1.split([num_move, data_5_1.size(0) - num_move])
            # label_5_1_move, label_5_1_base = label_5_1.split([num_move, label_5_1.size(0) - num_move])

            data_5_1_move, data_5_1_base = data_5_1.split([data_5_1.size(0)-1, 1])
            label_5_1_move, label_5_1_base = label_5_1.split([data_5_1.size(0)-1, 1])


            data4_new = torch.cat((data_4_0_base, data_4_1, data_5_1_move), 0)
            label4_new = torch.cat((label_4_0_base, label_4_1, label_5_1_move), 0)


            data5_new = torch.cat((data_5_1_base, data_5_0, data_4_0_move), 0)
            label5_new = torch.cat((label_5_1_base, label_5_0, label_4_0_move), 0)


            list_split_datasets_4 = list(split_datasets[4])
            list_split_datasets_5 = list(split_datasets[5])



            list_split_datasets_4[0] = torch.clone(data4_new)
            list_split_datasets_4[1] = torch.clone(label4_new)        

            list_split_datasets_5[0] = torch.clone(data5_new)
            list_split_datasets_5[1] = torch.clone(label5_new)

            split_datasets[4] = tuple(list_split_datasets_4)
            split_datasets[5] = tuple(list_split_datasets_5)

# -----------------------------------------------

        index_6_0 = split_datasets[6][1] == 0
        data_6_0 = split_datasets[6][0][index_6_0]
        data_6_1 = split_datasets[6][0][~index_6_0]
        label_6_0 = split_datasets[6][1][index_6_0]
        label_6_1 = split_datasets[6][1][~index_6_0]

        data_6_0_move, data_6_0_base = data_6_0.split([data_6_0.size(0)-1, 1])
        label_6_0_move, label_6_0_base = label_6_0.split([data_6_0.size(0)-1, 1])

        # data_6_0_move, data_6_0_base = data_6_0.split([num_move, data_6_0.size(0) - num_move])
        # label_6_0_move, label_6_0_base = label_6_0.split([num_move, label_6_0.size(0) - num_move])


        index_7_0 = split_datasets[7][1] == 0
        data_7_0 = split_datasets[7][0][index_7_0]
        data_7_1 = split_datasets[7][0][~index_7_0]
        label_7_0 = split_datasets[7][1][index_7_0]
        label_7_1 = split_datasets[7][1][~index_7_0]

        # data_7_1_move, data_7_1_base = data_7_1.split([num_move, data_7_1.size(0) - num_move])
        # label_7_1_move, label_7_1_base = label_7_1.split([num_move, label_7_1.size(0) - num_move])

        data_7_1_move, data_7_1_base = data_7_1.split([data_7_1.size(0)-1, 1])
        label_7_1_move, label_7_1_base = label_7_1.split([data_7_1.size(0)-1, 1])


        data6_new = torch.cat((data_6_0_base, data_6_1, data_7_1_move), 0)
        label6_new = torch.cat((label_6_0_base, label_6_1, label_7_1_move), 0)


        data7_new = torch.cat((data_7_1_base, data_7_0, data_6_0_move), 0)
        label7_new = torch.cat((label_7_1_base, label_7_0, label_6_0_move), 0)


        list_split_datasets_6 = list(split_datasets[6])
        list_split_datasets_7 = list(split_datasets[7])



        list_split_datasets_6[0] = torch.clone(data6_new)
        list_split_datasets_6[1] = torch.clone(label6_new)        

        list_split_datasets_7[0] = torch.clone(data7_new)
        list_split_datasets_7[1] = torch.clone(label7_new)

        split_datasets[6] = tuple(list_split_datasets_6)
        split_datasets[7] = tuple(list_split_datasets_7)


        # ---------------------------------------------------------------------------------------------------------------------------------


        index_8_0 = split_datasets[8][1] == 0
        data_8_0 = split_datasets[8][0][index_8_0]
        data_8_1 = split_datasets[8][0][~index_8_0]
        label_8_0 = split_datasets[8][1][index_8_0]
        label_8_1 = split_datasets[8][1][~index_8_0]

        data_8_0_move, data_8_0_base = data_8_0.split([data_8_0.size(0)-1, 1])
        label_8_0_move, label_8_0_base = label_8_0.split([data_8_0.size(0)-1, 1])

        # data_8_0_move, data_8_0_base = data_8_0.split([num_move+buff_move, data_8_0.size(0) - num_move -buff_move])
        # label_8_0_move, label_8_0_base = label_8_0.split([num_move+buff_move, label_8_0.size(0) - num_move-buff_move])

        index_9_0 = split_datasets[9][1] == 0
        data_9_0 = split_datasets[9][0][index_9_0]
        data_9_1 = split_datasets[9][0][~index_9_0]
        label_9_0 = split_datasets[9][1][index_9_0]
        label_9_1 = split_datasets[9][1][~index_9_0]

        data_9_1_move, data_9_1_base = data_9_1.split([data_9_1.size(0)-1, 1])
        label_9_1_move, label_9_1_base = label_9_1.split([data_9_1.size(0)-1, 1])

        # data_9_1_move, data_9_1_base = data_9_1.split([num_move+buff_move, data_9_1.size(0) - num_move-buff_move])
        # label_9_1_move, label_9_1_base = label_9_1.split([num_move+buff_move, label_9_1.size(0) - num_move-buff_move])

        data8_new = torch.cat((data_8_0_base, data_8_1, data_9_1_move), 0)
        label8_new = torch.cat((label_8_0_base, label_8_1, label_9_1_move), 0)


        data9_new = torch.cat((data_9_1_base, data_9_0, data_8_0_move), 0)
        label9_new = torch.cat((label_9_1_base, label_9_0, label_8_0_move), 0)


        list_split_datasets_8 = list(split_datasets[8])
        list_split_datasets_9 = list(split_datasets[9])



        list_split_datasets_8[0] = torch.clone(data8_new)
        list_split_datasets_8[1] = torch.clone(label8_new)        

        list_split_datasets_9[0] = torch.clone(data9_new)
        list_split_datasets_9[1] = torch.clone(label9_new)

        split_datasets[8] = tuple(list_split_datasets_8)
        split_datasets[9] = tuple(list_split_datasets_9)





        # 0  1  2  3  4  5  6  7  8  9
        # 10 11 12 13 14 15 16 17 18 19
        # datasets = list()
        # # Ti le data non-iid 1 1 1 1 1 1 0.5 0.5 2 2 
        # for i in range(num_clients):
        #     if 5 < i < 10:
        #         if i == 8:
        #             datasets.append( (torch.cat((split_datasets[i][0],split_datasets[16][0],split_datasets[17][0]), 0), 
        #                             torch.cat((split_datasets[i][1], split_datasets[16][1],split_datasets[17][1]), 0) ))
        #         elif i == 9:
        #             datasets.append( (torch.cat((split_datasets[i][0],split_datasets[18][0],split_datasets[19][0]), 0), 
        #                             torch.cat((split_datasets[i][1], split_datasets[18][1], split_datasets[19][1]), 0) ))
        #         else:
        #             datasets.append( (split_datasets[i][0], split_datasets[i][1]) )
        #     else:
        #         datasets.append( (torch.cat((split_datasets[i][0],split_datasets[i+10][0]), 0), 
        #                         torch.cat((split_datasets[i][1], split_datasets[i+10][1]), 0) ))



        # finalize bunches of xPlocal datasets


        

    if attack_mode in ['Label-Flipping']:
        datasets = list()
        # Label Flipping for 4 attackers
        for i in range(num_clients):
            if 0 < i < num_attackers+1:
                labels = split_datasets[i][1].cpu().detach().numpy()
                new_labels = np.array([abs(s-1) for s in labels])
                datasets.append((split_datasets[i][0], torch.Tensor(new_labels)))
            else:
                datasets.append((split_datasets[i][0], split_datasets[i][1]))
        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset)
            for local_dataset in datasets
            ]
        

    elif attack_mode in ['GAN']:
        if model_name == "CNN":
            GAN_dim = training_inputs.shape[2]
            g_param = torch.load('/home/haochu/Documents/PoisoningAttack/Results/CICToNIoT/CNN/FedGAN.pth',map_location=lambda x,y:x)
        else:
            GAN_dim = training_inputs.shape[2]*training_inputs.shape[3]
            g_param = torch.load('/home/haochu/Documents/PoisoningAttack/Results/CICToNIoT/LeNet/FedGAN.pth',map_location=lambda x,y:x)
        gan = Generator(GAN_dim, GAN_dim)
        gan.load_state_dict(g_param)
        datasets = list()
        for i in range(num_clients):
            if 0 < i < num_attackers+1:
                samples = split_datasets[i][0].cpu().detach().numpy()
                if model_name == "CNN":
                    samples = np.reshape(samples, (samples.shape[0], samples.shape[2]))
                else:
                    samples = np.reshape(samples, (samples.shape[0], samples.shape[2]*samples.shape[3]))
                # Original samples as Dataframe
                ori_samples = pd.DataFrame(samples, columns=columns)
                ori_samples['Label'] = split_datasets[i][1].cpu().detach().numpy()
                
                # Generated mutated samples by pretrain IDSGAN model
                mutated_samples, mutated_labels = gan.generate(gan, ori_samples, GAN_dim, BATCH_SIZE=1024)
                mutated_samples = mutated_samples.to_numpy()
                if model_name == "CNN":
                    mutated_samples = np.reshape(mutated_samples, (mutated_samples.shape[0], 1, mutated_samples.shape[1]))
                else:
                    mutated_samples = np.reshape(mutated_samples, (mutated_samples.shape[0], 1, 5, mutated_samples.shape[1]//5))
                datasets.append((torch.Tensor(mutated_samples), torch.Tensor(mutated_labels)))
            else:
                datasets.append((split_datasets[i][0], split_datasets[i][1]))
        
        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset)
            for local_dataset in datasets
            ]

    else:
        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset)
            for local_dataset in split_datasets
            ]
    
    shuffled_indices = torch.randperm(len(testing_inputs))
    testing_inputs = testing_inputs[shuffled_indices]
    testing_labels = torch.Tensor(testing_labels)[shuffled_indices]

    testing_dataset = list(
        zip(
            torch.split(torch.Tensor(testing_inputs), len(testing_inputs)), 
            torch.split(torch.Tensor(testing_labels), len(testing_labels))
            )
        )
    
    return local_datasets, CustomTensorDataset(testing_dataset[0])