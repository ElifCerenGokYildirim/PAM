import copy
import torch.nn.utils.prune as prune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from data_manager import DataManager
from model import TaskSpecificResNet
import logging

class SAM:
    def __init__(self, dataset_name, epochs, shuffle, seed, increment, freeze, lr, threshold, lamda, temperature, sparsity):
        self.data_manager = DataManager(dataset_name, shuffle, seed, increment, increment)
        self.known_classes = 0
        self.increment = increment
        self.total_classes = self.known_classes + self.increment
        self.current_task = self.known_classes // self.increment
        self.model = TaskSpecificResNet()
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.task_centroids = {}
        self.task_to_block_map = {}
        self.model.init_model(self.increment)
        self.lr = lr
        self.freeze = freeze
        self.threshold = threshold
        self.lamda = lamda
        self.temperature = temperature
        self.sparsity = sparsity
        self.it_prune = False

        self.train_accuracies = []
        self.test_accuracies = []
        self.number_of_parameters = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        logging.info("Training session started.")
        logging.info(f'known classes {self.known_classes}')
        logging.info(f'total classes {self.total_classes}')

        train_dataset = self.data_manager.get_dataset(indices=np.arange(self.known_classes, self.total_classes),
                                                      source='train',
                                                      mode='train')
        train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)

        new_task_centroid = self.find_task_centroid(train_loader)

        if self.current_task != 0:
            most_similar_task, distance, avg_distance = self.measure_task_similarity(new_task_centroid)
            similarity_threshold = self.threshold * avg_distance
            logging.info(f"current similarity threshold calculated as :, {similarity_threshold}")

            if distance < similarity_threshold:
                logging.info(f"Current task {self.current_task} is found similar with {most_similar_task}. "
                             f"It will reuse last block.")
                # Reuse the most similar task's block
                self.task_to_block_map[self.current_task] = self.task_to_block_map[most_similar_task]

                # Map current task to most similar task's block
                self.current_task_block = self.task_to_block_map[self.current_task]  # Reuse block
                # Apply smaller learning rate for regularization
                # self.lr = 0.0005

                # Freeze the teacher model
                teacher_model = copy.deepcopy(self.model)  # Deepcopy of the model to be used as the teacher
                teacher_model.eval()  # Set teacher model to evaluation mode to avoid gradient updates

            else:
                logging.info(f"Task {self.current_task} is found different and new last block is created.")

                # Create a new block for the current task and initialize weights from the most similar task
                best_similar_task = self.task_to_block_map[most_similar_task]
                #self.model.add_block(self.current_task, best_similar_task)
                # Create a new block for the current task and initialize with pretrained weights
                self.model.add_block(self.current_task)

                # Map current task to its own block
                self.task_to_block_map[self.current_task] = self.current_task
                self.current_task_block = self.current_task  # New block


            self.model.expand_fc(self.known_classes, self.total_classes)

        else:
            logging.info(f"New specialized block is initialized for {self.current_task}")
            self.model.add_block(self.current_task)

            # Map first task to its own block
            self.task_to_block_map[self.current_task] = self.current_task
            self.current_task_block = self.current_task

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        
        self.model.train()
        self.model.to(self.device)

        #self.prune_last_block(self.current_task_block, sparsity=self.sparsity)
        for epoch in range(self.epochs):
            correct_train = 0
            total_train = 0
            if epoch == 1:
                self.prune_last_block(self.current_task_block, sparsity=self.sparsity)
                self.it_prune=True
            for batch_idx, (_, data, target) in enumerate(train_loader):
                #if batch_idx ==7:
                    #self.prune_last_block(self.current_task_block, sparsity=self.sparsity)

                # Move data to the same device as the model
                target = target.type(torch.LongTensor)
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                if self.current_task > 0 and distance < similarity_threshold:
                    _, student_output = self.model(data, self.current_task_block)

                    logits = student_output
                    fake_targets = target - self.known_classes  # Adjust the targets for the new classes
                    classification_loss = F.cross_entropy(logits[:, self.known_classes:],
                                           fake_targets)  # Compute loss only on new classes
                    with torch.no_grad():
                        _, teacher_output = teacher_model(data, self.current_task_block)
                    distillation_loss = self.KD_loss(logits[:, :self.known_classes], teacher_output, self.temperature)
                    loss = classification_loss + self.lamda * distillation_loss
                else:
                    _, student_output = self.model(data, self.current_task_block)
                    logits = student_output
                    fake_targets = target - self.known_classes  # Adjust the targets for the new classes
                    loss = F.cross_entropy(logits[:, self.known_classes:],
                                                          fake_targets)  # Compute loss only on new classes

                loss.backward()
                self.model.classification_head.weight.grad[:self.known_classes] = 0
                self.model.classification_head.bias.grad[:self.known_classes] = 0
                self.optimizer.step()

                _, predicted = torch.max(logits, 1)
                correct_train += (predicted == target).sum().item()
                total_train += target.size(0)
                if self.it_prune:
                    self.prune_last_block(self.current_task_block, sparsity=self.sparsity)
            train_acc = 100 * correct_train / total_train
            self.train_accuracies.append(train_acc)
            logging.info(f'Train Accuracy: {train_acc:.2f}% for Epoch {epoch}')

        self.it_prune=False
        self.prune_last_block(self.current_task_block, sparsity=self.sparsity, before_train=False)
        self.task_centroids[self.current_task] = new_task_centroid
        logging.info(f"Centroid for Task {self.current_task} stored.")

    def after_task(self):
        self.known_classes = self.total_classes
        self.total_classes = self.known_classes + self.increment
        self.current_task = self.known_classes // self.increment
        # Count parameters in shared layers
        shared_params, classification_head_params = self._count_parameters()
        total_params = sum(self.number_of_parameters) + shared_params + classification_head_params
        logging.info(f'Total number of trainable parameters up until this task: {sum(self.number_of_parameters)}')
        logging.info(f'Number of shared and frozen parameter size: {shared_params}')
        logging.info(f'Number of parameters in classification layer: {classification_head_params}')
        logging.info(f'Total number of parameters: {total_params}')
        print('Total parameters used in this continual system is:', sum(self.number_of_parameters) + shared_params + classification_head_params)

    def _count_parameters(self):
        shared_params = 0
        classification_head_params = 0
        for param in self.model.shared_layers.parameters():
            shared_params += param.numel()

        for param in self.model.classification_head.parameters():
            classification_head_params += param.numel()
        return shared_params, classification_head_params

    def prune_last_block(self, task_id, sparsity, before_train=True):
        block = self.model.task_specific_blocks[str(task_id)]
        if before_train:
            # Prune the Conv2d layers in specialized last block
            for module in block.modules():
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=sparsity, n=1, dim=0)  # Prune filters (dim=0)
                    prune.remove(module, 'weight')  # Optionally remove the pruning reparameterization

        # Calculate the number remaining parameters after pruning
        if not before_train:
               non_zero_params_after = sum(param[param != 0].numel() for param in block.parameters() if param.requires_grad)
               self.number_of_parameters.append(non_zero_params_after)

    def find_task_centroid(self, data_loader):
        """Calculate and return the centroid (mean) of the task's feature representations."""
        resnet18 = models.resnet18(pretrained=True)
        common_layers = list(resnet18.children())[:-3]
        resnet18_common = torch.nn.Sequential(*common_layers)
        resnet18_common.to(self.device)
        resnet18_common.eval()
        feature_sum = None
        total_samples = 0

        with torch.no_grad():
            for _, data, target in data_loader:
                data = data.to(self.device)
                features = resnet18_common(data)  # Forward pass through frozen blocks
                if feature_sum is None:
                    feature_sum = torch.sum(features, dim=0)
                else:
                    feature_sum += torch.sum(features, dim=0)
                total_samples += features.size(0)

        centroid = feature_sum / total_samples
        return centroid

    def measure_task_similarity(self, new_task_centroid):
        """Measure similarity between the new task and previously learned tasks."""
        if not self.task_centroids:
            return None, float('inf')  # No previous tasks, no similarity
        min_distance = float('inf')
        most_similar_task = None
        previous_distances = []
        for task_id, centroid in self.task_centroids.items():
            distance = F.pairwise_distance(new_task_centroid.view(1, -1), centroid.view(1, -1), p=1).item()
            logging.info(f"Distance between task {task_id} and new task: {distance}")
            previous_distances.append(distance)

            if distance < min_distance:
                min_distance = distance
                most_similar_task = task_id

        avg_distance = sum(previous_distances) / len(previous_distances)
        return most_similar_task, min_distance, avg_distance

    def KD_loss(self, pred, soft, T):
        # Log-softmax over the predicted logits (current model)
        pred = torch.log_softmax(pred / T, dim=1)
        # Softmax over the logits from the teacher model (previous model)
        soft = torch.softmax(soft / T, dim=1)
        # Calculate KD loss by measuring KL-divergence
        return F.kl_div(pred, soft, reduction="batchmean")  # KL-divergence loss


    def evaluate_with_til_setup(self, task_id):
        self.model.eval()
        task_accs = []
        for task in range(task_id+1):
            test_dataset = self.data_manager.get_dataset(indices=np.arange(task*self.increment, (task+1)*self.increment),
                                                         source='test',
                                                         mode='test')

            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            block_id = self.task_to_block_map[task]

            correct = 0
            samples = 0

            with torch.no_grad():
                for _, data, target in test_loader:
                    target = target.type(torch.LongTensor)
                    data, target = data.to(self.device), target.to(self.device)
                    _, output = self.model(data, task_id=block_id)
                    _, predicted = torch.max(output, dim=1)
                    samples += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / samples
            self.test_accuracies.append(accuracy)
            task_accs.append(accuracy)
            logging.info(f'Task {task} Accuracy: {accuracy:.2f}%')

        incremental_acc = sum(task_accs) / len(task_accs)
        logging.info(f'Incremental Accuracy after Task {task_id}: {incremental_acc:.2f}%')
        return task_accs

    def evaluate_with_similarity(self, task_id):
        """
        Evaluate model using prototype-based task routing
        """
        self.model.eval()
        task_accs = []

        for current_task in range(task_id + 1):
            print('current task', current_task)
            # Prepare test dataset for the current task
            test_dataset = self.data_manager.get_dataset(
                indices=np.arange(current_task * self.increment, (current_task + 1) * self.increment),
                source='test',
                mode='test'
            )
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            new_task_centroid = self.find_task_centroid(test_loader)
            most_similar_task, distance, avg_distance = self.measure_task_similarity(new_task_centroid)
            logging.info(f'The most similar task to current one found during the test time: {most_similar_task}')

            correct = 0
            samples = 0

            with torch.no_grad():
                for _, data, target in test_loader:
                    target = target.type(torch.LongTensor)
                    data, target = data.to(self.device), target.to(self.device)
                    #new_task_centroid = self.find_task_centroid(test_loader)
                    #most_similar_task, distance, avg_distance = self.measure_task_similarity(new_task_centroid)
                    #logging.info(f'The most similar task to current one found during the test time: {most_similar_task}')
                    _, output = self.model(data, task_id=most_similar_task)
                    _, predicted = torch.max(output, dim=1)
                    samples += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / samples
            self.test_accuracies.append(accuracy)
            task_accs.append(accuracy)
            logging.info(f'Task {task_id} Accuracy: {accuracy:.2f}%')

        incremental_acc = sum(task_accs) / len(task_accs)
        logging.info(f'Incremental Accuracy after Task {task_id}: {incremental_acc:.2f}%')
        return task_accs

    def evaluate_with_confidence(self, task_id):
        self.model.eval()
        task_accs = []

        for current_task in range(task_id + 1):
            test_dataset = self.data_manager.get_dataset(
                indices=np.arange(current_task * self.increment, (current_task + 1) * self.increment),
                source='test',
                mode='test'
            )
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            correct = 0
            samples = 0

            with torch.no_grad():
                for _, data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    # Find best task WITHOUT using labels
                    best_task = None
                    max_confidence = -float('inf')

                    # Choose task based on model confidence
                    for potential_task in self.task_to_block_map.keys():
                        _, output = self.model(data, task_id=self.task_to_block_map[potential_task])
                        probabilities = torch.softmax(output, dim=1)
                        confidence = torch.max(probabilities, dim=1).values.mean().item()

                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_task = potential_task

                    # Run inference with selected task
                    _, output = self.model(data, task_id=self.task_to_block_map[best_task])
                    _, predicted = torch.max(output, dim=1)

                    # Now compute accuracy
                    correct += (predicted == target).sum().item()
                    samples += target.size(0)

                    logging.info(f"Selected task {best_task} for batch")

            accuracy = 100 * correct / samples
            task_accs.append(accuracy)
            logging.info(f'Task {current_task} Accuracy: {accuracy:.2f}%')

        incremental_acc = sum(task_accs) / len(task_accs)
        logging.info(f'Incremental Accuracy after Task {task_id}: {incremental_acc:.2f}%')
        return task_accs

    def find_top_k_similar_tasks(self, new_task_centroid, k=3):
        """Find k most similar tasks based on centroid distance."""
        distances = []
        for task_id, centroid in self.task_centroids.items():
            # Calculate L1 distance
            distance = F.pairwise_distance(new_task_centroid.view(1, -1), centroid.view(1, -1), p=1).item()
            distances.append((task_id, distance))
            logging.info(f"Distance between task {task_id} and new task: {distance}")

        # Sort by distance and return top k
        sorted_tasks = sorted(distances, key=lambda x: x[1])
        top_k_tasks = [task_id for task_id, _ in sorted_tasks[:k]]

        # Log the selected tasks and their distances
        for task_id, dist in sorted_tasks[:k]:
            logging.info(f"Selected task {task_id} with distance {dist}")

        return top_k_tasks


