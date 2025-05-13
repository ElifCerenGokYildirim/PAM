import copy
import torch.nn as nn
import torchvision.models as models


class TaskSpecificResNet(nn.Module):
    def __init__(self):
        super(TaskSpecificResNet, self).__init__()

    def init_model(self, increment):
        self.resnet = models.resnet152(pretrained=True)
        self.task_specific_blocks = nn.ModuleDict()

        # Define shared layers (frozen)
        self.shared_layers = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3
        )

        # Freeze shared layers
        for param in self.shared_layers.parameters():
            param.requires_grad = False

        # Task-specific blocks start from layer4 (unfrozen)
        self.last_block_template = nn.Sequential(self.resnet.layer4,
                                                 nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Flatten())

        self.classification_head = self.generate_fc(2048, increment)

    def add_block(self, task_id, similar_task_id=None):
        if similar_task_id is not None:
            # Copy the weights from the most similar task's block
            self.task_specific_blocks[str(task_id)] = copy.deepcopy(self.task_specific_blocks[str(similar_task_id)])
            print(f"New block for Task {task_id} initialized with weights from Task {similar_task_id}.")
        else:
            # Initialize with the pretrained weights (default behavior)
            self.task_specific_blocks[str(task_id)] = copy.deepcopy(self.last_block_template)
            print(f"New block for Task {task_id} initialized with pretrained weights.")

    def generate_fc(self, in_dim, out_dim):
        return nn.Linear(in_dim, out_dim)

    def expand_fc(self, known_classes, total_classes):
        new_classification_head = self.generate_fc(2048, total_classes)
        # Copy old weights
        new_classification_head.weight.data[:known_classes] = self.classification_head.weight.data
        new_classification_head.bias.data[:known_classes] = self.classification_head.bias.data
        self.classification_head = new_classification_head
        print(f"Classification head expanded to {total_classes} classes.")

    def freeze(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        return self

    def forward(self, x, task_id):
        # Pass through shared layers
        x = self.shared_layers(x)
        # Task-specific last block
        #if task_id is not None:
        #print("Current keys in task_specific_blocks:", self.task_specific_blocks.keys())
        features = self.task_specific_blocks[str(task_id)](x)
        logits = self.classification_head(features)
        return features, logits
