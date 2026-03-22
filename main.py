import logging
import random
import torch
from pam import PAM
from toolkit import set_logger
import numpy as np
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    dataset_name = 'cifar100'
    shuffle = True
    seed = 1993
    increment = 5 # Number of classes per task
    num_tasks = 20
    epoch = 25
    freeze = False
    lr = 0.001
    threshold = 0.1
    distill_type = "Logit"
    lamda = 0.2
    temperature = 1
    sparsity = 0.96
    model = 'resnet152'

    set_seed(seed)

    set_logger(dataset_name, num_tasks, seed, freeze, lr, threshold, distill_type, lamda, temperature, sparsity, epoch, model)

    sam = SAM(dataset_name, epoch, shuffle, seed, increment, freeze, lr, threshold, lamda, temperature, sparsity)
    # Initialize accuracy matrix
    acc_matrix = np.zeros((num_tasks, num_tasks), dtype=float)
    test_times = []  # <--- Store per-task test times here

    for task in range(num_tasks):
        sam.train()
        # Time only the evaluation step
        start = time.time()
        task_accuracies = sam.evaluate_with_confidence(task)
        end = time.time()
        elapsed = end - start
        test_times.append(elapsed)
        logging.info(f"Test time after learning up until task {task}: {elapsed:.4f} seconds")
        #task_accuracies = sam.evaluate_with_similarity(task)
        #task_accuracies = sam.evaluate_with_til_setup(task)
        for eval_task, acc in enumerate(task_accuracies):
            acc_matrix[task, eval_task] = acc  # Store accuracy in the matrix

        sam.after_task()
    logging.info("\nFinal Incremental Accuracy Matrix:")
    logging.info(acc_matrix)



if __name__ == "__main__":
    main()
