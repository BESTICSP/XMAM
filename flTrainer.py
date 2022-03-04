import numpy as np
import logging
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from model import *
from dataLoader import *
from defenders import *
from attackers import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
import pdb
from scipy.stats.mstats import hmean
import sys

from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def train(model, data_loader, device, criterion, optimizer):

    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        loss = criterion(output, batch_y) # cross entropy loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            logger.info("loss: {}".format(loss))
    return model

def malicious_train(model, global_model_pre, whole_data_loader, clean_data_loader, poison_data_loader, device, criterion, optimizer,
                    attack_mode="none", scaling=10, pgd_eps=5e-2, untargeted_type='sign-flipping'):

    model.train()

    ################################################################## attack mode
    if attack_mode == "none":
        for batch_idx, (batch_x, batch_y) in enumerate(whole_data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y) # cross entropy loss
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                logger.info("loss: {}".format(loss))

    elif attack_mode == "stealthy":
        ### title:Analyzing federated learning through an adversarial lens
        for  poison_data, clean_data in zip(poison_data_loader, clean_data_loader):
            poison_data[0], clean_data[0] = poison_data[0].to(device), clean_data[0].to(device)
            poison_data[1], clean_data[1] = poison_data[1].to(device), clean_data[1].to(device)
            optimizer.zero_grad()
            output = model(poison_data[0])
            loss1 = criterion(output, poison_data[1]) # cross entropy loss
            output = model(clean_data[0])
            loss2 = criterion(output, clean_data[1])  # cross entropy loss

            avg_update_pre = parameters_to_vector(list(global_model_pre.parameters())) - parameters_to_vector(list(global_model_pre.parameters()))
            mine_update_now = parameters_to_vector(list(model.parameters())) - parameters_to_vector(list(global_model_pre.parameters()))
            loss = loss1 + loss2 + 10**(-4)*torch.norm(mine_update_now - avg_update_pre)**2

            loss.backward()
            optimizer.step()

            logger.info("loss: {}".format(loss))

    elif attack_mode == "pgd":
        ### l2_projection
        project_frequency = 10
        eps = pgd_eps
        for batch_idx, (batch_x, batch_y) in enumerate(whole_data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)  # cross entropy loss
            loss.backward()
            optimizer.step()
            w = list(model.parameters())
            w_vec = parameters_to_vector(w)
            model_original_vec = parameters_to_vector(list(global_model_pre.parameters()))
            # make sure you project on last iteration otherwise, high LR pushes you really far
            if (batch_idx % project_frequency == 0 or batch_idx == len(whole_data_loader) - 1) and (
                    torch.norm(w_vec - model_original_vec) > eps):
                # project back into norm ball
                w_proj_vec = eps * (w_vec - model_original_vec) / torch.norm(
                    w_vec - model_original_vec) + model_original_vec
                # plug w_proj back into model
                vector_to_parameters(w_proj_vec, w)
            logger.info("loss: {}".format(loss))



    elif attack_mode == "replacement":
        whole_aggregator = []
        for p_index, p in enumerate(model.parameters()):
            params_aggregator = torch.zeros(p.size()).to(device)
            params_aggregator = list(global_model_pre.parameters())[p_index].data + \
                                scaling * (list(model.parameters())[p_index].data -
                                      list(global_model_pre.parameters())[p_index].data)
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(model.parameters()):
            p.data = whole_aggregator[param_index]


    ###################################################################### untargeted attacks
    if untargeted_type == 'sign-flipping':
        whole_aggregator = []
        for p_index, p in enumerate(model.parameters()):
            params_aggregator = list(global_model_pre.parameters())[p_index].data - \
                                10*(list(model.parameters())[p_index].data -
                                   list(global_model_pre.parameters())[p_index].data)
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(model.parameters()):
            p.data = whole_aggregator[param_index]

    elif untargeted_type == 'same-value':
        whole_aggregator = []
        for p_index, p in enumerate(model.parameters()):
            params_aggregator = list(global_model_pre.parameters())[p_index].data + \
                                1*torch.sign(list(model.parameters())[p_index].data -
                                   list(global_model_pre.parameters())[p_index].data)
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(model.parameters()):
            p.data = whole_aggregator[param_index]

    return model

def test_model(model, data_loader, device, print_perform=False):
    model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return accuracy_score(y_true.cpu(), y_predict.cpu())

#### fed_avg
def fed_avg_aggregator(net_list, global_model_pre, device, model="lenet", num_class=10):

    net_avg = global_model_pre
    #### observe parameters
    # net_glo_vec = vectorize_net(global_model_pre)
    # print("{}   :  {}".format(-1, net_glo_vec[10000:10010]))
    # for i in range(len(net_list)):
    #     net_vec = vectorize_net(net_list[i])
    #     print("{}   :  {}".format(i, net_vec[10000:10010]))

    whole_aggregator = []

    for p_index, p in enumerate(net_list[0].parameters()):
        # initial
        params_aggregator = torch.zeros(p.size()).to(device)
        for net_index, net in enumerate(net_list):
            params_aggregator = params_aggregator + 1/len(net_list) * list(net.parameters())[p_index].data
        whole_aggregator.append(params_aggregator)

    for param_index, p in enumerate(net_avg.parameters()):
        p.data = whole_aggregator[param_index]
    return net_avg


class ParameterContainer:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def run(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class FederatedLearningTrainer(ParameterContainer):
    def __init__(self, arguments=None, *args, **kwargs):
        self.net_avg = arguments['net_avg']
        self.partition_strategy = arguments['partition_strategy']
        self.dir_parameter = arguments['dir_parameter']
        self.net_dataidx_map = arguments['net_dataidx_map']
        self.num_nets = arguments['num_nets']
        self.part_nets_per_round = arguments['part_nets_per_round']
        self.fl_round = arguments['fl_round']
        self.local_training_epoch = arguments['local_training_epoch']
        self.malicious_local_training_epoch = arguments['malicious_local_training_epoch']
        self.args_lr = arguments['args_lr']
        self.args_gamma = arguments['args_gamma']
        self.batch_size = arguments['batch_size']
        self.device = arguments['device']
        self.dataname = arguments["dataname"]
        self.num_class = arguments["num_class"]
        self.datadir = arguments["datadir"]
        self.model = arguments["model"]
        self.load_premodel = arguments["load_premodel"]
        self.save_model = arguments["save_model"]
        self.client_select = arguments["client_select"]
        self.test_data_ori_loader = arguments["test_data_ori_loader"]
        self.test_data_backdoor_loader = arguments["test_data_backdoor_loader"]
        self.criterion = nn.CrossEntropyLoss()
        self.malicious_ratio = arguments["malicious_ratio"]
        self.trigger_label = arguments["trigger_label"]
        self.semantic_label = arguments["semantic_label"]
        self.poisoned_portion = arguments["poisoned_portion"]
        self.attack_mode = arguments["attack_mode"]
        self.pgd_eps = arguments["pgd_eps"]
        self.backdoor_type = arguments["backdoor_type"]
        self.model_scaling = arguments["model_scaling"]
        self.untargeted_type = arguments["untargeted_type"]
        self.defense_method = arguments["defense_method"]

    def run(self):

        fl_iter_list = []
        main_task_acc = []
        backdoor_task_acc = []
        client_chosen = []
        train_loader_list = []

        train_data, test_data = load_init_data(dataname=self.dataname, datadir=self.datadir)
        xmam_data = copy.deepcopy(train_data)

        ################################################################ distribute data to clients before training
        if self.backdoor_type == 'semantic':
            dataidxs = self.net_dataidx_map[9999]
            clean_idx = self.net_dataidx_map[99991]
            poison_idx = self.net_dataidx_map[99992]
            train_data_loader_semantic = create_train_data_loader_semantic(train_data, self.batch_size, dataidxs,
                                                              clean_idx, poison_idx)
        if self.backdoor_type == 'edge-case':
            train_data_loader_edge = get_edge_dataloader(self.datadir, self.batch_size)

        for c in range(self.num_nets):
            if c < self.malicious_ratio * self.num_nets:
                if self.backdoor_type == 'none':
                    dataidxs = self.net_dataidx_map[c]
                    train_data_loader = create_train_data_loader(self.dataname, train_data, self.trigger_label,
                                                                 self.poisoned_portion, self.batch_size, dataidxs,
                                                                 malicious=False)
                elif self.backdoor_type == 'trigger':
                    dataidxs = self.net_dataidx_map[c]
                    train_data_loader  = create_train_data_loader(self.dataname, train_data, self.trigger_label,
                                                             self.poisoned_portion, self.batch_size, dataidxs,
                                                             malicious=True)


                elif self.backdoor_type == 'semantic':
                    train_data_loader = train_data_loader_semantic

                elif self.backdoor_type == 'edge-case':
                    train_data_loader = train_data_loader_edge

                if self.untargeted_type == 'label-flipping':
                    dataidxs = self.net_dataidx_map[c]
                    train_data_loader = create_train_data_loader_lf(train_data, self.batch_size, dataidxs, self.num_class)

                train_loader_list.append(train_data_loader)
            else:
                dataidxs = self.net_dataidx_map[c]
                train_data_loader = create_train_data_loader(self.dataname, train_data, self.trigger_label,
                                                             self.poisoned_portion, self.batch_size, dataidxs,
                                                             malicious=False)
                train_loader_list.append(train_data_loader)

        ########################################################################################## multi-round training
        for flr in range(1, self.fl_round+1):

            norm_diff_collector = []  # for NDC-adaptive
            g_user_indices = []  # for krum and multi-krum
            malicious_num = 0  # for krum and multi-krum
            nets_list = [i for i in range(self.num_nets)]
            # output the information about data number of selected clients

            if self.client_select == 'fix-pool':
                selected_node_indices = np.random.choice(nets_list, size=self.part_nets_per_round, replace=False)
            elif self.client_select == 'fix-frequency':
                selected_node_mali = np.random.choice(nets_list[ :int(self.num_nets * self.malicious_ratio)],
                                            size=int(self.part_nets_per_round * self.malicious_ratio), replace=False)
                selected_node_mali = selected_node_mali.tolist()
                selected_node_benign = np.random.choice(nets_list[int(self.num_nets * self.malicious_ratio): ],
                                            size=int(self.part_nets_per_round * (1-self.malicious_ratio)), replace=False)
                # selected_node_benign = np.array([0])
                selected_node_benign = selected_node_benign.tolist()
                selected_node_mali.extend(selected_node_benign)
                selected_node_indices = selected_node_mali

            num_data_points = [len(self.net_dataidx_map[i]) for i in selected_node_indices]
            net_data_number = [num_data_points[i] for i in range(self.part_nets_per_round)]
            logger.info("client data number: {}, FL round: {}".format(net_data_number, flr))

            # we need to reconstruct the net list at the beginning
            net_list = [copy.deepcopy(self.net_avg) for _ in range(self.part_nets_per_round)]
            logger.info("################## Starting fl round: {}".format(flr))

            ### for stealthy attack, we reserve previous global model
            if flr == 1:
                global_model_pre = copy.deepcopy(self.net_avg)
            else:
                pass

            # start the FL process

            for net_idx, net in enumerate(net_list):

                global_user_idx = selected_node_indices[net_idx]
                if global_user_idx < self.malicious_ratio * self.num_nets:

                    logger.info("$malicious$ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))
                    for e in range(1, self.malicious_local_training_epoch + 1):
                        optimizer = optim.SGD(net.parameters(), lr=self.args_lr * self.args_gamma ** (flr - 1),
                                              momentum=0.9,
                                              weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion
                        for param_group in optimizer.param_groups:
                            logger.info("Effective lr in fl round: {} is {}".format(flr, param_group['lr']))

                        if not self.backdoor_type == 'none':
                            malicious_train(net, global_model_pre, train_loader_list[global_user_idx][0],
                                            train_loader_list[global_user_idx][1],
                                            train_loader_list[global_user_idx][2], self.device,
                                            self.criterion, optimizer, self.attack_mode, self.model_scaling,
                                            self.pgd_eps, self.untargeted_type)
                        else:
                            malicious_train(net, global_model_pre, train_loader_list[global_user_idx],
                                            train_loader_list[global_user_idx],
                                            train_loader_list[global_user_idx], self.device,
                                            self.criterion, optimizer, self.attack_mode, self.model_scaling,
                                            self.pgd_eps, self.untargeted_type)
                    malicious_num += 1
                    g_user_indices.append(global_user_idx)
                else:

                    logger.info("@benign@ Working on client: {}, which is Global user: {}".format(net_idx, global_user_idx))
                    for e in range(1, self.local_training_epoch + 1):
                        optimizer = optim.SGD(net.parameters(), lr=self.args_lr * self.args_gamma ** (flr - 1),
                                              momentum=0.9,
                                              weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion
                        for param_group in optimizer.param_groups:
                            logger.info("Effective lr in fl round: {} is {}".format(flr, param_group['lr']))

                        train(net, train_loader_list[global_user_idx], self.device, self.criterion, optimizer)

                    g_user_indices.append(global_user_idx)

                ### calculate the norm difference between global model pre and the updated benign client model for DNC's norm-bound
                vec_global_model_pre = parameters_to_vector(list(global_model_pre.parameters()))
                vec_updated_client_model = parameters_to_vector(list(net.parameters()))
                norm_diff = torch.norm(vec_updated_client_model - vec_global_model_pre)
                logger.info("the norm difference between global model pre and the updated benign client model: {}".format(norm_diff))
                norm_diff_collector.append(norm_diff.item())


            ########################################################################################## attack process
            if self.untargeted_type == 'krum-attack':
                self.attacker = krum_attack()
                net_list = self.attacker.exec(client_models=net_list, malicious_num=malicious_num,
                    global_model_pre=self.net_avg, expertise='full-knowledge', num_workers=self.part_nets_per_round,
                        num_dps=net_data_number, g_user_indices=g_user_indices, device=self.device)

            elif self.untargeted_type == 'xmam-attack':
                self.attacker = xmam_attack()
                ### generate an All-Ones matrix
                if self.dataname == 'mnist':
                    xmam_data.data = torch.ones_like(train_data.data[0:1])
                elif self.dataname in ('cifar10', 'cifar100'):
                    # xmam_data.data = np.ones_like(train_data.data[0:1])
                    xmam_data.data = train_data.data[0:1]

                xmam_data.targets = train_data.targets[0:1]
                x_ray_loader = create_train_data_loader(self.dataname, xmam_data, self.trigger_label,
                                                        self.poisoned_portion, self.batch_size, [0], malicious=False)
                net_list = self.attacker.exec(client_models=net_list, malicious_num=malicious_num,
                                              global_model_pre=self.net_avg, expertise='full-knowledge',
                                              x_ray_loader=x_ray_loader,
                                              num_workers=self.part_nets_per_round, num_dps=net_data_number,
                                              g_user_indices=g_user_indices, device=self.device)

            ########################################################################################## defense process
            if self.defense_method == "none":
                self.defender = None
                chosens = 'none'

            elif self.defense_method == "krum":
                self.defender = Krum(mode='krum', num_workers=self.part_nets_per_round, num_adv=malicious_num)
                net_list, net_freq, chosens = self.defender.exec(client_models=net_list, global_model_pre=self.net_avg, num_dps=net_data_number,
                                                        g_user_indices=g_user_indices, device=self.device)


            elif self.defense_method == "multi-krum":
                if malicious_num > 0:
                    self.defender = Krum(mode='multi-krum', num_workers=self.part_nets_per_round, num_adv=malicious_num)
                    net_list, net_freq, chosens = self.defender.exec(client_models=net_list, global_model_pre=self.net_avg, num_dps=net_data_number,
                                                       g_user_indices=g_user_indices, device=self.device)

                else:
                    chosens = g_user_indices
            elif self.defense_method == "xmam":
                self.defender = XMAM()
                ### generate an All-Ones matrix
                if self.dataname == 'mnist':
                    xmam_data.data = torch.ones_like(train_data.data[0:1])
                elif self.dataname in ('cifar10', 'cifar100'):
                    # xmam_data.data = np.ones_like(train_data.data[0:1])
                    xmam_data.data = train_data.data[0:1]

                xmam_data.targets = train_data.targets[0:1]
                x_ray_loader = create_train_data_loader(self.dataname, xmam_data, self.trigger_label,
                             self.poisoned_portion, self.batch_size, [0], malicious=False)
                net_list, chosens = self.defender.exec(client_models=net_list, x_ray_loader=train_loader_list[0], global_model_pre=self.net_avg,
                                                g_user_indices=g_user_indices, device=self.device, malicious_ratio=self.malicious_ratio)

            elif self.defense_method == "ndc":
                chosens = 'none'
                logger.info("@@@ Nom Diff Collector Mean: {}".format(np.mean(norm_diff_collector)))
                self.defender = WeightDiffClippingDefense(norm_bound=np.mean(norm_diff_collector))
                for net_idx, net in enumerate(net_list):
                    self.defender.exec(client_model=net, global_model=global_model_pre)

            elif self.defense_method == "rsa":
                chosens = 'none'
                self.defender = RSA()
                self.defender.exec(client_model=net_list, global_model=global_model_pre, flround=flr)

            elif self.defense_method == "rfa":
                chosens = 'none'
                self.defender = RFA()
                net_list = self.defender.exec(client_model=net_list, maxiter=5, eps=0.1, ftol=1e-5, device=self.device)

            elif self.defense_method == "weak-dp":
                chosens = 'none'
                self.defender = AddNoise(stddev=0.0005)
                for net_idx, net in enumerate(net_list):
                    self.defender.exec(client_model=net, device=self.device)
            else:
                # NotImplementedError("Unsupported defense method !")
                pass

            ########################################################################################################

            #################################### after local training periods and defence process, we fedavg the nets
            global_model_pre = self.net_avg

            self.net_avg = fed_avg_aggregator(net_list, global_model_pre, device=self.device, model=self.model, num_class=self.num_class)

            v = torch.nn.utils.parameters_to_vector(self.net_avg.parameters())
            logger.info("############ Averaged Model : Norm {}".format(torch.norm(v)))

            logger.info("Measuring the accuracy of the averaged global model, FL round: {} ...".format(flr))

            overall_acc = test_model(self.net_avg, self.test_data_ori_loader, self.device, print_perform=False)
            logger.info("=====Main task test accuracy=====: {}".format(overall_acc))

            backdoor_acc = test_model(self.net_avg, self.test_data_backdoor_loader, self.device, print_perform=False)
            logger.info("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))

            if self.save_model == True:
                # if (overall_acc > 0.8) or flr == 2:
                if flr == 50:
                    torch.save(self.net_avg.state_dict(), "savedModel/mnist_lenet.pt")
                    # sys.exit()

            fl_iter_list.append(flr)
            main_task_acc.append(overall_acc)
            backdoor_task_acc.append(backdoor_acc)
            client_chosen.append(chosens)


        #################################################################################### save result to .csv
        df = pd.DataFrame({'fl_iter': fl_iter_list,
                            'main_task_acc': main_task_acc,
                            'backdoor_task_acc': backdoor_task_acc,
                            'the chosen ones': client_chosen,
                            })

        results_filename = '101010__1-{}_2-{}_3-{}_4-{}_5-{}_6-{}_7-{}_8-{}_9-{}_10-{}_11-{}_12-{}_13-{}_14-{}_15-{}_16-{}' \
                           '_17-{}_18-{}_19-{}_20-{}_21-{}_22-{}'.format(
            self.dataname,  #1
            self.partition_strategy,  #2
            self.dir_parameter,  #3
            self.args_lr,  #4
            self.fl_round,  #5
            self.local_training_epoch,  #6
            self.malicious_local_training_epoch,  #7
            self.malicious_ratio,  #8
            self.part_nets_per_round,  #9
            self.num_nets,  #10
            self.poisoned_portion,  #11
            self.trigger_label,  #12
            self.attack_mode,  #13
            self.defense_method,  #14
            self.model,  #15
            self.load_premodel,  #16
            self.backdoor_type,  #17
            self.untargeted_type, #18
            self.model_scaling,   #19
            self.client_select,   #20
            self.pgd_eps,   #21
            self.semantic_label,   #22
        )

        df.to_csv('result/{}.csv'.format(results_filename), index=False)
        logger.info("Wrote accuracy results to: {}".format(results_filename))



