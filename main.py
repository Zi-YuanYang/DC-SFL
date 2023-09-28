import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import copy
import logging
import glob
import re

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from phe import paillier

from Server_Model import *
from datasets import trainset_loader
from torch.utils.data import DataLoader
import wandb

wandb.login(key='xxxxx')

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--model", type=str, default='Uformer', help='Uformer or NB_Uformer')
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-8)
parser.add_argument("--n_block", type=int, default=50)
parser.add_argument("--n_cpu", type=int, default=0)

###federated paras
parser.add_argument("--num_clients", type=int, default=4, help='Number of local clients')
parser.add_argument("--communication", type=int, default=500, help='Number of communications')
parser.add_argument("--epochs", type=int, default=1, help="Number of local training")
parser.add_argument("--mode", type=str, default='Encrypted_DCSFL', help="SL|sfl|Yang")
parser.add_argument("--mu", type=float, default=1e-4, help="the weight of fedprox")

###file paras
parser.add_argument('--checkpoint_interval', type=int, default=50)
parser.add_argument("--model_save_path", type=str, default="saved_models/")
parser.add_argument("--data_path", type=str, default="../../Dataset/FedData/Mat Data/small_data/")
parser.add_argument('--log_path', type=str, default="./log/")

parser.add_argument('--dp',type = int, default=1, help="Diffi Pri Swit")
parser.add_argument('--sigma',type = float, default=0.01, help="The Noise Stren")
parser.add_argument('--length',type=int,default=512,help="the length of public key")
parser.add_argument('--seed',type=int,default=42)
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)



cuda = True

wandb.init(project='SplFed',
           name=opt.mode + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
           job_type="training")


def get_logger(logger_name, log_file, level=logging.INFO):
    ## Read the Codes by yourself
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")  # RECORD Time
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    vlog = logging.getLogger(logger_name)
    vlog.setLevel(level)
    vlog.addHandler(fileHandler)

    return vlog


def create_logger(indx):
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

    time_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    mode_name = opt.mode
    model_name = opt.model

    if not os.path.exists((opt.log_path + mode_name + '_' + model_name + '_' + time_name)):
        os.makedirs((opt.log_path + mode_name + '_' + model_name + '_' + time_name))

    log_file = opt.log_path + mode_name + '_' + model_name + '_' + time_name + '/' + 'client' + '_' + str(indx) + '.log'
    logger = get_logger('NB', log_file)
    return logger


def my_collate_test(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    prj_data = [item[1] for item in batch]
    res_name = [item[2] for item in batch]
    option = torch.stack([item[3] for item in batch], 0)
    feature = torch.stack([item[4] for item in batch], 0)
    return input_data, prj_data, res_name, option, feature


def my_collate(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    label_data = torch.stack([item[1] for item in batch], 0)
    prj_data = [item[2] for item in batch]
    option = torch.stack([item[3] for item in batch], 0)
    feature = torch.stack([item[4] for item in batch], 0)
    return input_data, label_data, prj_data, option, feature


def Dataset():
    ### Build Dataset

    src_dataset_1 = DataLoader(trainset_loader(opt.data_path + "geometry_1"),
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    src_dataset_2 = DataLoader(trainset_loader(opt.data_path + "geometry_2"),
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    src_dataset_3 = DataLoader(trainset_loader(opt.data_path + "geometry_3"),
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
    src_dataset_4 = DataLoader(trainset_loader(opt.data_path + "geometry_4"),
                               batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)

    dataloaders = []
    dataloaders.append(src_dataset_1)
    dataloaders.append(src_dataset_2)
    dataloaders.append(src_dataset_3)
    dataloaders.append(src_dataset_4)

    return dataloaders

def encrypt_vector(public_key, parameters):
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters

def decrypt_vector(private_key, parameters):
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    return parameters


def add_noise(parameters, dp, device, sigma=0.02):
    noise = None
    # 不加噪声
    if dp == 0:
        return parameters
    # 拉普拉斯噪声
    elif dp == 1:
        noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape)).to(device)
    # 高斯噪声
    else:
        noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, sigma)

    return parameters.add_(noise)

class net():
    def __init__(self):
        mode_name = opt.mode
        #        model_name = opt.model
        self.path = opt.model_save_path + mode_name  # +'_'+model_nme
        self.path = opt.model_save_path + mode_name  # +'_'+model_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        ##Dataset
        self.train_datas = Dataset()

        ##Training Settings
        self.start = 0
        self.epoch = opt.epochs
        self.com = opt.communication
        self.client_num = opt.num_clients
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = [create_logger(idx) for idx in range(opt.num_clients)]
        self.client_weights = [1 / self.client_num for i in range(self.client_num)]
        self.alpha = 0
        ##Model
        self.client_model_1 = Client_Model1().to(self.device)
        self.client_model_2 = Client_Model2().to(self.device)
        self.server_model = Server_Model().to(self.device)

        self.client_models_1 = [copy.deepcopy(self.client_model_1) for idx in range(self.client_num)]
        self.client_models_2 = [copy.deepcopy(self.client_model_2) for idx in range(self.client_num)]
        self.server_models = [copy.deepcopy(self.server_model) for idx in range(self.client_num)]

        self.client_1_optimizers = [torch.optim.Adam(self.client_models_1[idx].parameters(),
                                                     lr=opt.lr, weight_decay=opt.weight_decay)
                                    for idx in range(self.client_num)]
        self.client_2_optimizers = [torch.optim.Adam(self.client_models_2[idx].parameters(),
                                                     lr=opt.lr, weight_decay=opt.weight_decay)
                                    for idx in range(self.client_num)]
        self.server_optimizer = torch.optim.Adam(self.server_model.parameters(),
                                                 lr=opt.lr, weight_decay=opt.weight_decay)
        self.server_optimizers = [torch.optim.Adam(self.server_models[idx].parameters(),
                                                   lr=opt.lr, weight_decay=opt.weight_decay)
                                  for idx in range(self.client_num)]

        ## Previous Models
        self.pre_server_model = copy.deepcopy(self.server_model)
        self.pre_client_model_1 = copy.deepcopy(self.client_model_1)
        self.pre_client_model_2 = copy.deepcopy(self.client_model_2)

        self.temp_server_model = copy.deepcopy(self.server_model)
        self.temp_client_model_1 = copy.deepcopy(self.client_model_1)
        self.temp_client_model_2 = copy.deepcopy(self.client_model_2)

        self.server_optimizer_prox = torch.optim.Adam(self.temp_server_model.parameters(),
                                                      lr=opt.lr, weight_decay=opt.weight_decay)
        self.client1_optimizer_prox = torch.optim.Adam(self.temp_client_model_1.parameters(),
                                                       lr=opt.lr, weight_decay=opt.weight_decay)
        self.client2_optimizer_prox = torch.optim.Adam(self.temp_client_model_2.parameters(),
                                                       lr=opt.lr, weight_decay=opt.weight_decay)

        self.dp = opt.dp
        self.sigma = opt.sigma
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=opt.length)
        print('------Initlization Finished------')

    def train(self):

        dataset_length = len(self.train_datas[0])
        iter_index = [1, 1, 1, 1]

        print('------Start Training------')
        for com_iter in range(self.start, self.com):

            sum_parameters_cli_1 = None
            parameters_shape_cli_1 = None
            # sum_parameters_body = None
            # parameters_shape_body = None
            sum_parameters_cli_2 = None
            parameters_shape_cli_2 = None

            for epoch in range(self.epoch):
                for i_wkr in range(self.client_num):
                    for batch_index in range(dataset_length):

                        input_data, label_data, _, _, _ = next(iter(self.train_datas[i_wkr]))
                        input_data = input_data.to(self.device)
                        label_data = label_data.to(self.device)

                        self.client_1_optimizers[i_wkr].zero_grad()
                        out_1 = self.client_models_1[i_wkr](input_data)
                        # if dp == 1:
                        #     out_1 = add_noise(out_1, self.dp, self.device)
                        client_feature_1 = out_1.clone().detach().requires_grad_(True)

                        self.server_optimizers[i_wkr].zero_grad()
                        server_out = self.server_models[i_wkr](client_feature_1)
                        # server_out = add_noise(server_out, self.dp, self.device)
                        server_model_feature = server_out.clone().detach().requires_grad_(True)
                               
                        self.client_2_optimizers[i_wkr].zero_grad()
                        final_output = self.client_models_2[i_wkr](server_model_feature, input_data)

                        loss = self.loss(final_output, label_data)
                        loss.backward()
                        self.client_2_optimizers[i_wkr].step()

                        dfx_server = server_model_feature.grad.clone().detach()
                        server_out.backward(dfx_server)
                        self.server_optimizers[i_wkr].step()

                        dfx_client_1 = client_feature_1.grad.clone().detach()
                        out_1.backward(dfx_client_1)
                        self.client_1_optimizers[i_wkr].step()

                        print(
                            "Com Round: %d | Worker id: %d | [Epoch %d/%d] [Batch %d/%d]: [loss: %f]"
                            % (com_iter, i_wkr, epoch + 1, self.epoch, batch_index + 1, len(self.train_datas[i_wkr]),
                               loss.item())
                        )

                        self.logger[i_wkr].info(
                            'worker id: [{}] com:[{}] epoch: [{}] batch [{}/{}], loss:{:.7f}'.format(
                                i_wkr, com_iter, epoch, batch_index + 1, len(self.train_datas[i_wkr]), loss.item())
                        )

                        wandb.log({"loss_" + str(i_wkr): loss.item(), "local iteration": iter_index[i_wkr]})
                        iter_index[i_wkr] += 1

                    print('Start Encrypt')
                    ### Encrypt
                    if sum_parameters_cli_1 is None:
                        sum_parameters_cli_1 = {}
                        parameters_shape_cli_1 = {}
                        for key, var in self.client_models_1[i_wkr].state_dict().items():
                            sum_parameters_cli_1[key] = var
                            parameters_shape_cli_1[key] = var.shape
                            sum_parameters_cli_1[key] = add_noise(sum_parameters_cli_1[key],self.dp,self.device,self.sigma)#.cpu().numpy()
                            sum_parameters_cli_1[key] = encrypt_vector(self.public_key,sum_parameters_cli_1[key])
                    else:
                        for key in sum_parameters_cli_1:
                            sum_parameters_cli_1[key] = np.add(sum_parameters_cli_1[key], encrypt_vector(self.public_key, add_noise(self.client_models_1[i_wkr].state_dict()[key], self.dp, self.device,self.sigma)))  ### Encryption
                            # sum_parameters_cli_1[key] = np.add(sum_parameters_cli_1[key], add_noise(self.client_models_1[i_wkr].state_dict()[key], self.dp, self.device,self.sigma).cpu().numpy())   #### Differential Privacy

                    # if sum_parameters_body is None:
                    #     sum_parameters_body = {}
                    #     parameters_shape_body = {}
                    #     for key, var in self.server_models[i_wkr].state_dict().items():
                    #         sum_parameters_body[key] = var
                    #         parameters_shape_body[key] = var.shape
                    #         sum_parameters_body[key] = add_noise(sum_parameters_body[key],self.dp,self.device,self.sigma)
                    #         sum_parameters_body[key] = encrypt_vector(self.public_key,sum_parameters_body[key])
                    # else:
                        # for key in sum_parameters_body:
                        #     sum_parameters_body[key] = np.add(sum_parameters_body[key], encrypt_vector(self.public_key, add_noise(self.server_models[i_wkr].state_dict().items()[key], self.dp, self.device,self.sigma)))
                    # print('2')

                    if sum_parameters_cli_2 is None:
                        sum_parameters_cli_2 = {}
                        parameters_shape_cli_2 = {}
                        for key, var in self.client_models_2[i_wkr].state_dict().items():
                            sum_parameters_cli_2[key] = var
                            parameters_shape_cli_2[key] = var.shape
                            sum_parameters_cli_2[key] = add_noise(sum_parameters_cli_2[key],self.dp,self.device,self.sigma)#.cpu().numpy()
                            sum_parameters_cli_2[key] = encrypt_vector(self.public_key,sum_parameters_cli_2[key])
                    else:
                        for key in sum_parameters_cli_2:
                            sum_parameters_cli_2[key] = np.add(sum_parameters_cli_2[key], encrypt_vector(self.public_key, add_noise(self.client_models_2[i_wkr].state_dict()[key], self.dp, self.device,self.sigma)))
                            # sum_parameters_cli_2[key] = np.add(sum_parameters_cli_2[key], add_noise(self.client_models_2[i_wkr].state_dict()[key], self.dp, self.device,self.sigma).cpu().numpy())

                ### Decrypt & Com
                self.server_communication()

                for key in self.client_model_1.state_dict().keys():
                    sum_parameters_cli_1[key] = decrypt_vector(self.private_key, sum_parameters_cli_1[key])
                    sum_parameters_cli_1[key] = torch.reshape(torch.Tensor(sum_parameters_cli_1[key]), parameters_shape_cli_1[key])
                    self.client_model_1.state_dict()[key].data.copy_(sum_parameters_cli_1[key] / self.client_num)

                    for client_idx in range(self.client_num):
                        self.client_models_1[client_idx].state_dict()[key].data.copy_(self.client_model_1.state_dict()[key])

                for key in self.client_model_2.state_dict().keys():
                    sum_parameters_cli_2[key] = decrypt_vector(self.private_key, sum_parameters_cli_2[key])
                    sum_parameters_cli_2[key] = torch.reshape(torch.Tensor(sum_parameters_cli_2[key]), parameters_shape_cli_2[key])
                    self.client_model_2.state_dict()[key].data.copy_(sum_parameters_cli_2[key] / self.client_num)

                    for client_idx in range(self.client_num):
                        self.client_models_2[client_idx].state_dict()[key].data.copy_(self.client_model_2.state_dict()[key])

                
                if com_iter > 0:
                    self.DWCS(com_iter - 1)

                if opt.checkpoint_interval != -1 and (com_iter + 1) % opt.checkpoint_interval == 0:
                    torch.save(self.server_model.state_dict(), '%s/model_commu_%04d.pth' % (self.path, com_iter + 1))
                    for check_id in range(self.client_num):
                        torch.save(self.client_models_1[check_id].state_dict(),
                                   '%s/model_worker_id(%04d)_commu_%04d_client1.pth' % (
                                   self.path, check_id, com_iter + 1))
                        torch.save(self.client_models_2[check_id].state_dict(),
                                   '%s/model_worker_id(%04d)_commu_%04d_client2.pth' % (
                                   self.path, check_id, com_iter + 1))
                        torch.save(self.server_models[check_id].state_dict(),
                                   '%s/model_worker_id(%04d)_commu_%04d_server.pth' % (
                                   self.path, check_id, com_iter + 1))

    def client1_communication(self):
        client_weights = [1 / self.client_num for i in range(self.client_num)]
        with torch.no_grad():
            for key in self.client_model_1.state_dict().keys():
                temp = torch.zeros_like(self.client_model_1.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(client_weights)):
                    temp += self.client_weights[client_idx] * self.client_models_1[client_idx].state_dict()[key]
                self.client_model_1.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    self.client_models_1[client_idx].state_dict()[key].data.copy_(self.client_model_1.state_dict()[key])

    def client2_communication(self):
        client_weights = [1 / self.client_num for i in range(self.client_num)]
        with torch.no_grad():
            for key in self.client_model_2.state_dict().keys():
                temp = torch.zeros_like(self.client_model_2.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(client_weights)):
                    temp += self.client_weights[client_idx] * self.client_models_2[client_idx].state_dict()[key]
                self.client_model_2.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    self.client_models_2[client_idx].state_dict()[key].data.copy_(self.client_model_2.state_dict()[key])

    def server_communication(self):
        client_weights = [1 / self.client_num for i in range(self.client_num)]

        with torch.no_grad():
            for key in self.server_model.state_dict().keys():
                temp = torch.zeros_like(self.server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(client_weights)):
                    temp += self.client_weights[client_idx] * self.server_models[client_idx].state_dict()[key]
                self.server_model.state_dict()[key].data.copy_(temp)

                for client_idx in range(len(client_weights)):
                    self.server_models[client_idx].state_dict()[key].data.copy_(self.server_model.state_dict()[key])

    def Alpha_aggre(self, epoch):
        ## Calculate the weight
        alpha_upper = 0.99
        self.alpha = min(1 - 1 / (epoch + 1), alpha_upper)
        ### The first client model
        for param, pre_param in zip(self.client_model_1, self.pre_client_model_1):
            param.data.mul_(self.alpha).add_(1 - self.alpha, pre_param.data)
        ### The second client model
        for param, pre_param in zip(self.client_model_2, self.pre_client_model_2):
            param.data.mul_(self.alpha).add_(1 - self.alpha, pre_param.data)
        ### The third client model
        for param, pre_param in zip(self.server_model, self.pre_server_model):
            param.data.mul_(self.alpha).add_(1 - self.alpha, pre_param.data)
        ### Change pre models
        self.Change_premodels()

    def DWCS(self, epoch):

        ## Give Temp Value.
        self.Change_Temp_Model()

        ## Calculate the weight
        up_alpha = 0.99
        self.alpha = min(1 - 1 / (epoch + 1), up_alpha)
        ## Server_Prox
        self.server_optimizer_prox.zero_grad()
        w_diff = torch.tensor(0., device=self.device)
        for param, pre_param in zip(self.temp_server_model.parameters(), self.pre_server_model.parameters()):
            w_diff += torch.pow(torch.norm(param - pre_param), 2)
        loss = opt.mu / 2. * w_diff
        loss.backward()
        self.server_optimizer_prox.step()
        for param, pre_param in zip(self.server_model.parameters(), self.temp_server_model.parameters()):
            param.data.mul_(1 - self.alpha).add_(self.alpha, pre_param.data)

        ## Client_1_Prox
        self.client1_optimizer_prox.zero_grad()
        w_diff = torch.tensor(0., device=self.device)
        for param, pre_param in zip(self.client_model_1.parameters(), self.temp_client_model_1.parameters()):
            w_diff += torch.pow(torch.norm(param - pre_param), 2)
        loss = opt.mu / 2. * w_diff
        loss.backward()
        self.client1_optimizer_prox.step()
        for param, pre_param in zip(self.client_model_1.parameters(), self.temp_client_model_1.parameters()):
            param.data.mul_(1 - self.alpha).add_(self.alpha, pre_param.data)

        ## Client_2_Prox
        self.client2_optimizer_prox.zero_grad()
        w_diff = torch.tensor(0., device=self.device)
        for param, pre_param in zip(self.client_model_2.parameters(), self.temp_client_model_2.parameters()):
            w_diff += torch.pow(torch.norm(param - pre_param), 2)
        loss = opt.mu / 2. * w_diff
        loss.backward()
        self.client2_optimizer_prox.step()
        for param, pre_param in zip(self.client_model_2.parameters(), self.temp_client_model_2.parameters()):
            param.data.mul_(1 - self.alpha).add_(self.alpha, pre_param.data)

        ### Change pre models
        self.Change_premodels()

    def Change_Temp_Model(self):
        for key in self.client_model_1.state_dict().keys():
            self.temp_client_model_1.state_dict()[key].data.copy_(self.client_model_1.state_dict()[key])
        for key in self.client_model_2.state_dict().keys():
            self.temp_client_model_2.state_dict()[key].data.copy_(self.client_model_2.state_dict()[key])
        for key in self.server_model.state_dict().keys():
            self.temp_server_model.state_dict()[key].data.copy_(self.server_model.state_dict()[key])

    def Change_premodels(self):

        ### Change the Client 1 Model
        for key in self.client_model_1.state_dict().keys():
            self.pre_client_model_1.state_dict()[key].data.copy_(self.client_model_1.state_dict()[key])
            for client_idx in range(self.client_num):
                self.client_models_1[client_idx].state_dict()[key].data.copy_(self.client_model_1.state_dict()[key])

        ### Change the Client 2 Model
        for key in self.client_model_2.state_dict().keys():
            self.pre_client_model_2.state_dict()[key].data.copy_(self.client_model_2.state_dict()[key])
            for client_idx in range(self.client_num):
                self.client_models_2[client_idx].state_dict()[key].data.copy_(self.client_model_2.state_dict()[key])

                ### Change the Server Model
        for key in self.server_model.state_dict().keys():
            self.pre_server_model.state_dict()[key].data.copy_(self.server_model.state_dict()[key])
            for client_idx in range(self.client_num):
                self.server_models[client_idx].state_dict()[key].data.copy_(self.server_model.state_dict()[key])


if __name__ == '__main__':
    network = net()
    network.train()
