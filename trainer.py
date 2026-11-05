import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, init_wandb
import os
import numpy as np


def train(args):
    args = copy.deepcopy(args)
    seed_list = args["seed"]
    if isinstance(seed_list, int):
        seed_list = [seed_list]
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = copy.deepcopy(device)
        _train(args)


def _train(args):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    run = init_wandb(args)

    try:
        _set_random(args["seed"])
        _set_device(args)
        print_args(args)

        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
        
        args["nb_classes"] = data_manager.nb_classes # update args
        args["nb_tasks"] = data_manager.nb_tasks
        model = factory.get_model(args["model_name"], args)

        cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
        cnn_matrix, nme_matrix = [], []

        for task in range(data_manager.nb_tasks):
            logging.info("All params: {}".format(count_parameters(model._network)))
            logging.info(
                "Trainable params: {}".format(count_parameters(model._network, True))
            )
            model.incremental_train(data_manager)
            cnn_accy, nme_accy = model.eval_task()
            model.after_task()

            if nme_accy is not None:
                logging.info("CNN: {}".format(cnn_accy["grouped"]))
                logging.info("NME: {}".format(nme_accy["grouped"]))

                cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
                cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
                cnn_matrix.append(cnn_values)

                nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
                nme_values = [nme_accy["grouped"][key] for key in nme_keys]
                nme_matrix.append(nme_values)

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])

                nme_curve["top1"].append(nme_accy["top1"])
                nme_curve["top5"].append(nme_accy["top5"])

                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
                logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
                logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

                avg_cnn = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
                avg_nme = sum(nme_curve["top1"]) / len(nme_curve["top1"])

                print('Average Accuracy (CNN):', avg_cnn)
                print('Average Accuracy (NME):', avg_nme)

                logging.info("Average Accuracy (CNN): {}".format(avg_cnn))
                logging.info("Average Accuracy (NME): {}".format(avg_nme))
            else:
                logging.info("No NME accuracy.")
                logging.info("CNN: {}".format(cnn_accy["grouped"]))

                cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
                cnn_matrix.append(cnn_values)

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])

                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

                avg_cnn = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
                print('Average Accuracy (CNN):', avg_cnn)
                logging.info("Average Accuracy (CNN): {} \n".format(avg_cnn))

            if run is not None:
                import wandb
                total_params = count_parameters(model._network)
                trainable_params = count_parameters(model._network, True)
                avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
                metrics = {
                    "task": task,
                    "eval/top1_acc": cnn_accy["top1"],
                    "eval/top5_acc": cnn_accy["top5"],
                    "eval/avg_acc": avg_acc,
                    "params/total": total_params,
                    "params/trainable": trainable_params,
                }
                if nme_accy is not None:
                    avg_nme = sum(nme_curve["top1"]) / len(nme_curve["top1"])
                    metrics.update({
                        "eval/nme_top1_acc": nme_accy["top1"],
                        "eval/nme_top5_acc": nme_accy["top5"],
                        "eval/nme_avg_acc": avg_nme,
                    })
                wandb.log(metrics, step=task)

        if 'print_forget' in args.keys() and args['print_forget'] is True:
            if len(cnn_matrix) > 0:
                np_acctable = np.zeros([task + 1, task + 1])
                for idxx, line in enumerate(cnn_matrix):
                    idxy = len(line)
                    np_acctable[idxx, :idxy] = np.array(line)
                np_acctable = np_acctable.T
                cnn_forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
                print('Accuracy Matrix (CNN):')
                print(np_acctable)
                logging.info('Forgetting (CNN): {}'.format(cnn_forgetting))
                if run is not None:
                    import wandb
                    wandb.log({"eval/forgetting": float(cnn_forgetting)})
                    wandb.summary["eval/forgetting"] = float(cnn_forgetting)

            if len(nme_matrix) > 0:
                np_acctable = np.zeros([task + 1, task + 1])
                for idxx, line in enumerate(nme_matrix):
                    idxy = len(line)
                    np_acctable[idxx, :idxy] = np.array(line)
                np_acctable = np_acctable.T
                nme_forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
                print('Accuracy Matrix (NME):')
                print(np_acctable)
                logging.info('Forgetting (NME): {}'.format(nme_forgetting))
                if run is not None:
                    import wandb
                    wandb.log({"eval/nme_forgetting": float(nme_forgetting)})
                    wandb.summary["eval/nme_forgetting"] = float(nme_forgetting)

        if run is not None and len(cnn_curve["top1"]) > 0:
            import wandb
            wandb.summary["eval/avg_acc"] = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            wandb.summary["eval/top1_acc"] = cnn_curve["top1"][-1]

    finally:
        if run is not None:
            import wandb
            wandb.finish()


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if str(device) == "-1":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
