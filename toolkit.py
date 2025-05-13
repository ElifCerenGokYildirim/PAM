import logging
import os
import datetime
import numpy as np
import torch
import json
from enum import Enum

from matplotlib import pyplot as plt


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def set_logger(dataset, num_tasks, seed, freeze, lr, threshold, distill_type, lamda, temperature, pruning, epoch, model):
    """Set up logging configuration"""
    base_log_dir = os.path.expanduser("~/logs")  # Logs will be created in ~/logs
    os.makedirs(base_log_dir, exist_ok=True)
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    log_dir = f"logs/{dataset}/num_tasks{num_tasks}_seed{seed}_Freeze={freeze}_lr={lr}_threshold={threshold}_distill_type={distill_type}_lamda{lamda}_temperature{temperature}_pruning{pruning}epoch{epoch}_model{model}"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)  # Ensure the directory exists
    logging.basicConfig(filename=f"{log_dir}.log", level=logging.INFO,  # Log to file only
                        format="%(message)s")

def plot_accuracy(epochs, incremental_accuracies):
    epochs = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, incremental_accuracies, label='Inc Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show()

def accuracy(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
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

