from defenders import *

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()


def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight_diff[index_bias:index_bias + p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()


class Attack:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class krum_attack(Attack):

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, malicious_num, global_model_pre, expertise, num_workers, num_dps, g_user_indices, device, *args, **kwargs):

        s = copy.deepcopy(global_model_pre)
        if expertise == 'full-knowledge':
            whole_aggregator = []
            for p_index, p in enumerate(global_model_pre.parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                for net_index, net in enumerate(client_models):
                    params_aggregator = params_aggregator + torch.sign(list(net.parameters())[p_index].data -
                                                                list(global_model_pre.parameters())[p_index].data)
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(s.parameters()):
                p.data = whole_aggregator[param_index]
        elif expertise == 'partial-knowledge':
            whole_aggregator = []
            for p_index, p in enumerate(global_model_pre.parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                for net_index, net in enumerate(client_models[0:malicious_num]):
                    params_aggregator = params_aggregator + torch.sign(list(net.parameters())[p_index].data -
                                                                       list(global_model_pre.parameters())[
                                                                           p_index].data)
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(s.parameters()):
                p.data = whole_aggregator[param_index]

        lamuda = 0.2
        chosens = malicious_num
        byzantine_leader = copy.deepcopy(global_model_pre)
        while(chosens >= malicious_num and lamuda>1e-10):
            lamuda = lamuda / 2
            print("====lamuda====", lamuda)
            whole_aggregator = []
            for p_index, p in enumerate(byzantine_leader.parameters()):
                params_aggregator = list(global_model_pre.parameters())[p_index].data - \
                                    lamuda * list(s.parameters())[p_index].data
                whole_aggregator.append(params_aggregator)
            for param_index, p in enumerate(byzantine_leader.parameters()):
                p.data = whole_aggregator[param_index]

            for i in range(malicious_num):
                client_models[i] = byzantine_leader
            defender = Krum(mode='krum', num_workers=num_workers, num_adv=malicious_num)
            net_list, net_freq, chosens = defender.exec(client_models=client_models, global_model_pre=global_model_pre, num_dps=num_dps,
                                                                 g_user_indices=g_user_indices, device=device)

        return client_models


class xmam_attack(Attack):

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, malicious_num, global_model_pre, expertise, x_ray_loader, num_workers, num_dps, g_user_indices, device, *args, **kwargs):

        s = copy.deepcopy(global_model_pre)
        if expertise == 'full-knowledge':
            whole_aggregator = []
            for p_index, p in enumerate(global_model_pre.parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                for net_index, net in enumerate(client_models[malicious_num:]):
                    params_aggregator = params_aggregator + torch.sign(list(net.parameters())[p_index].data -
                                                                list(global_model_pre.parameters())[p_index].data)
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(s.parameters()):
                p.data = whole_aggregator[param_index]
        elif expertise == 'partial-knowledge':
            whole_aggregator = []
            for p_index, p in enumerate(global_model_pre.parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                for net_index, net in enumerate(client_models[0:malicious_num]):
                    params_aggregator = params_aggregator + torch.sign(list(net.parameters())[p_index].data -
                                                                       list(global_model_pre.parameters())[
                                                                           p_index].data)
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(s.parameters()):
                p.data = whole_aggregator[param_index]

        lamuda = 0.2
        chosens = [malicious_num]
        byzantine_leader = copy.deepcopy(global_model_pre)
        while(not 0 in chosens and lamuda>1e-10):
            lamuda = lamuda / 2
            print("====lamuda====", lamuda)
            whole_aggregator = []
            for p_index, p in enumerate(byzantine_leader.parameters()):
                params_aggregator = list(global_model_pre.parameters())[p_index].data - \
                                    lamuda * list(s.parameters())[p_index].data
                whole_aggregator.append(params_aggregator)
            for param_index, p in enumerate(byzantine_leader.parameters()):
                p.data = whole_aggregator[param_index]

            for i in range(malicious_num):
                client_models[i] = byzantine_leader
            self.defender = XMAM()
            net_list, chosens = self.defender.exec(client_models=client_models, x_ray_loader=x_ray_loader,
                                                   global_model_pre=global_model_pre,
                                                   g_user_indices=g_user_indices, device=device)
            print(chosens)

        return client_models







