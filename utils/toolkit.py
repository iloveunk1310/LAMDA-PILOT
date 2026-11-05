import os
import subprocess
import numpy as np
import torch
from collections import OrderedDict
import copy


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy, for initial classes
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    # for incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def state_dict_to_vector(state_dict, remove_keys=[]) -> torch.Tensor:
    shared_state_dict = copy.deepcopy(state_dict)
    shared_state_dict_keys = list(shared_state_dict.keys())
    for key in remove_keys:
        for _key in shared_state_dict_keys:
            if key in _key:
                del shared_state_dict[_key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    """
    Load vector into state_dict, except the keys in `remove_keys`.
    """
    removed_keys = []
    reference_dict = copy.deepcopy(state_dict)
    reference_dict_keys = list(reference_dict.keys())
    for key in remove_keys:
        for _key in reference_dict_keys:
            if key in _key:
                removed_keys.append(_key)
                del reference_dict[_key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    return sorted_reference_dict


def get_git_commit():
    """
    Retrieve current git commit hash if available.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def init_wandb(args):
    """
    Initialize a Weights & Biases run with standardized metadata.
    Only imports and initializes wandb if args.get("use_wandb") is True.
    """
    if not args.get("use_wandb", False):
        return None

    import wandb

    model_name = args.get("model_name", "unknown")
    dataset = args.get("dataset", "unknown")
    init_cls = 0 if args.get("init_cls") == args.get("increment") else args.get("init_cls", 0)
    increment = args.get("increment", 0)
    seed = args.get("seed", 0)
    backbone_type = args.get("backbone_type", "unknown")

    run_name = f"PILOT_{model_name}_{dataset}_b{init_cls}inc{increment}_s{seed}"
    group = args.get("wandb_group") or "LAMDA-PILOT"
    project = args.get("wandb_project") or "Group7-ML-Lab"
    entity = args.get("wandb_entity") or "nnm-hcmus"

    tags = [
        "repo:LAMDA-PILOT",
        f"method:{model_name}",
        f"dataset:{dataset}",
        f"backbone:{backbone_type}",
    ]
    custom_tags = args.get("wandb_tags")
    if custom_tags:
        if isinstance(custom_tags, list):
            tags.extend(custom_tags)
        else:
            tags.append(str(custom_tags))

    # Prepare serializable config dict
    config = {}
    for k, v in args.items():
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            config[k] = v
        else:
            config[k] = str(v)

    config["repo_name"] = "LAMDA-PILOT"
    config["git_commit"] = get_git_commit()

    init_kwargs = {
        "project": project,
        "group": group,
        "name": run_name,
        "tags": tags,
        "config": config,
        "reinit": "finish_previous",
    }
    if entity:
        init_kwargs["entity"] = entity
    if args.get("wandb_mode"):
        init_kwargs["mode"] = args["wandb_mode"]

    run = wandb.init(**init_kwargs)
    return run
