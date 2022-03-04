import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import flTrainerForEnsem
import flTrainerForNormal
import copy
from model.ensembleModel import ensembleModel
from model import normalForMnist, normalForCifar10
from model.vgg import get_vgg_model
from dataLoader import *
import torchvision

def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='parameter board')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.98, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--local_training_epoch', type=int, default=1, help='number of local training epochs')
    parser.add_argument('--malicious_local_training_epoch', type=int, default=1, help='number of malicious local training epochs')
    parser.add_argument('--num_nets', type=int, default=1, help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=1, help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=10, help='total number of FL round to conduct')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--dataname', type=str, default='cifar10', help='dataset to use during the training process')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--datadir', type=str, default='./dataset/', help='the directory of dataset')
    parser.add_argument('--partition_strategy', type=str, default='homo', help='dataset iid(homo) or non-iid(hetero-dir)')
    parser.add_argument('--dir_parameter', type=float, default=0.5, help='the parameter of dirichlet distribution')
    parser.add_argument('--model', type=str, default='ensemForCifar10', help='model to use during the training process')
    parser.add_argument('--load_premodel', type=bool_string, default=False, help='whether load the pre-model in begining')
    parser.add_argument('--save_model', type=bool_string, default=True, help='whether save the intermediate model')
    parser.add_argument('--client_select', type=str, default='fix-pool', help='the strategy for PS to select client: fix-frequency|fix-pool')

    # parameters for backdoor attacker
    parser.add_argument('--malicious_ratio', type=float, default=1, help='the ratio of malicious clients')
    parser.add_argument('--trigger_label', type=int, default=7, help='The NO. of trigger label (int, range from 0 to 9, default: 0)')
    parser.add_argument('--semantic_label', type=int, default=2, help='The NO. of semantic label (int, range from 0 to 9, default: 2)')
    parser.add_argument('--poisoned_portion', type=float, default=0.5, help='posioning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--attack_mode', type=str, default="none", help='attack method used: none|stealthy|pgd|replacement')
    parser.add_argument('--pgd_eps', type=float, default=5e-2, help='the eps of pgd')
    parser.add_argument('--backdoor_type', type=str, default="trigger", help='backdoor type used: none|trigger|semantic|edge-case|')
    parser.add_argument('--model_scaling', type=float, default=1, help='model replacement technology')

    # parameters for untargeted attacker
    parser.add_argument('--untargeted_type', type=str, default="none", help='untargeted type used: none|label-flipping|sign-flipping|same-value|')

    # parameters for defenders
    parser.add_argument('--defense_method', type=str, default="none",help='defense method used: none|krum|multi-krum|xmam|ndc|rsa|rfa|weak-dp|rlr|har')

    #############################################################################
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    device = torch.device(args.device if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    ###################################################################################### if ensemble networks
    if args.model == "ensemForMnist":

        if args.load_premodel == True:
            with open("savedModel/mnist_ensem_backdoored.pt", "rb") as ckpt_file:
                args.ensemModel = torch.load(ckpt_file)
            print("Loading pre-model successfully ...")
        else:
            ensemModelClass = ensembleModel(ensemForSet='mnist')
            G = ensemModelClass.Generator().to(device)
            R = ensemModelClass.Reconstructor().to(device)
            C = nn.ModuleDict({
                'cr': ensemModelClass.Classifier().to(device),
                'ci': ensemModelClass.Classifier().to(device)
            })
            D = nn.ModuleDict({
                'cr': ensemModelClass.Disentangler().to(device), 'ci': ensemModelClass.Disentangler().to(device)})

            args.ensemModel = {'G':G, 'D':D, 'C':C, 'R':R}

    elif args.model == "ensemForCifar10":

        if args.load_premodel == True:
            with open("savedModel/cifar10_ensem_backdoored.pt", "rb") as ckpt_file:
                args.ensemModel = torch.load(ckpt_file)
            print("Loading pre-model successfully ...")
        else:
            ensemModelClass = ensembleModel(ensemForSet='cifar10')
            G = ensemModelClass.Generator().to(device)
            R = ensemModelClass.Reconstructor().to(device)
            C = nn.ModuleDict({
                'cr': ensemModelClass.Classifier().to(device),
                'ci': ensemModelClass.Classifier().to(device)
            })
            D = nn.ModuleDict({
                'cr': ensemModelClass.Disentangler().to(device), 'ci': ensemModelClass.Disentangler().to(device)})
            args.ensemModel = {'G': G, 'D': D, 'C': C, 'R': R}



    if args.model == "normalForMnist":
        if args.load_premodel==True:
            net_avg = normalForMnist.normalNet().to(device)
            with open("savedModel/mnist_lenet_nobias.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            print("Loading pre-model successfully ...")
        else:
            net_avg = normalForMnist.normalNet().to(device)
    elif args.model == "normalForCifar10":
        if args.load_premodel==True:
            net_avg = normalForCifar10.normalNet().to(device)
            with open("savedModel/cifar10_vgg9_backdoored.pt", "rb") as ckpt_file:
                    ckpt_state_dict = torch.load(ckpt_file, map_location=device)
            net_avg.load_state_dict(ckpt_state_dict)
            print("Loading pre-model successfully ...")
        else:
            # net_avg = normalForCifar10.normalNet().to(device)
            net_avg = get_vgg_model(args.model, args.num_class)

    ############################################################################ adjust data distribution
    if args.backdoor_type in ('none', 'trigger'):
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'semantic':
        net_dataidx_map = partition_data_semantic(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)
    elif args.backdoor_type == 'edge-case':
        net_dataidx_map = partition_data(args.dataname, './dataset', args.partition_strategy, args.num_nets,
                                         args.dir_parameter)

    ########################################################################################## load dataset
    train_data, test_data = load_init_data(dataname=args.dataname, datadir=args.datadir)

    ######################################################################################### create data loader
    if args.backdoor_type == 'none':
        test_data_ori_loader, _ = create_test_data_loader(args.dataname, test_data, args.trigger_label,
                                                    args.batch_size)
        test_data_backdoor_loader = test_data_ori_loader
    elif args.backdoor_type == 'trigger':
        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader(args.dataname, test_data, args.trigger_label,
                                                     args.batch_size)
    elif args.backdoor_type == 'semantic':
        with open('./backdoorDataset/green_car_transformed_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        print("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = args.semantic_label * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # green car -> label as bird

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,
                                                                                           args.batch_size)
    elif args.backdoor_type == 'edge-case':
        with open('./backdoorDataset/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_greencar_dataset_test = pickle.load(test_f)

        print("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
        sampled_targets_array_test = 9 * np.ones((saved_greencar_dataset_test.shape[0],), dtype=int)  # southwest airplane -> label as truck

        semantic_testset = copy.deepcopy(test_data)
        semantic_testset.data = saved_greencar_dataset_test
        semantic_testset.targets = sampled_targets_array_test

        test_data_ori_loader, test_data_backdoor_loader = create_test_data_loader_semantic(test_data, semantic_testset,
                                                                                           args.batch_size)

    # print("Test the model performance on the entire task before FL process ... ")
    # overall_acc = test_model(net_avg, test_data_ori_loader, device, print_perform=True)
    # print("Test the model performance on the backdoor task before FL process ... ")
    # backdoor_acc = test_model(net_avg, test_data_backdoor_loader, device, print_perform=False)
    #
    # print("=====Main task test accuracy=====: {}".format(overall_acc))
    # print("=====Backdoor task test accuracy=====: {}".format(backdoor_acc))




    if args.model in ['ensemForMnist', 'ensemForCifar10']:
        arguments = {
            # "net_avg": net_avg,
            'ensemModel' : args.ensemModel,
            "partition_strategy": args.partition_strategy,
            "dir_parameter": args.dir_parameter,
            "net_dataidx_map": net_dataidx_map,
            "num_nets": args.num_nets,
            "dataname": args.dataname,
            "num_class": args.num_class,
            "datadir": args.datadir,
            "model": args.model,
            "load_premodel": args.load_premodel,
            "save_model": args.save_model,
            "client_select": args.client_select,
            "part_nets_per_round": args.part_nets_per_round,
            "fl_round": args.fl_round,
            "local_training_epoch": args.local_training_epoch,
            "malicious_local_training_epoch": args.malicious_local_training_epoch,
            "args_lr": args.lr,
            "args_gamma": args.gamma,
            "batch_size": args.batch_size,
            "device": device,
            "test_data_ori_loader": test_data_ori_loader,
            "test_data_backdoor_loader": test_data_backdoor_loader,
            "malicious_ratio": args.malicious_ratio,
            "trigger_label": args.trigger_label,
            "semantic_label": args.semantic_label,
            "poisoned_portion": args.poisoned_portion,
            "attack_mode": args.attack_mode,
            "pgd_eps": args.pgd_eps,
            "backdoor_type": args.backdoor_type,
            "model_scaling": args.model_scaling,
            "untargeted_type": args.untargeted_type,
            "defense_method": args.defense_method,
        }
        fl_trainer = flTrainerForEnsem.FederatedLearningTrainer(arguments=arguments)
        fl_trainer.run()

    elif args.model in ['normalForMnist', 'normalForCifar10']:
        arguments = {
            "net_avg": net_avg,
            "partition_strategy": args.partition_strategy,
            "dir_parameter": args.dir_parameter,
            "net_dataidx_map": net_dataidx_map,
            "num_nets": args.num_nets,
            "dataname": args.dataname,
            "num_class": args.num_class,
            "datadir": args.datadir,
            "model": args.model,
            "load_premodel": args.load_premodel,
            "save_model": args.save_model,
            "client_select": args.client_select,
            "part_nets_per_round": args.part_nets_per_round,
            "fl_round": args.fl_round,
            "local_training_epoch": args.local_training_epoch,
            "malicious_local_training_epoch": args.malicious_local_training_epoch,
            "args_lr": args.lr,
            "args_gamma": args.gamma,
            "batch_size": args.batch_size,
            "device": device,
            "test_data_ori_loader": test_data_ori_loader,
            "test_data_backdoor_loader": test_data_backdoor_loader,
            "malicious_ratio": args.malicious_ratio,
            "trigger_label": args.trigger_label,
            "semantic_label": args.semantic_label,
            "poisoned_portion": args.poisoned_portion,
            "attack_mode": args.attack_mode,
            "pgd_eps": args.pgd_eps,
            "backdoor_type": args.backdoor_type,
            "model_scaling": args.model_scaling,
            "untargeted_type": args.untargeted_type,
            "defense_method": args.defense_method,
        }

        fl_trainer = flTrainerForNormal.FederatedLearningTrainer(arguments=arguments)
        fl_trainer.run()
