"""
Typhoon Intensity Estimation ML Pipeline - Model Architectures
"""

import torch.nn as nn


class TyphoonMLP(nn.Module):
    """Multi-layer perceptron for typhoon intensity estimation.

    Architecture:
    - Flatten 240x240 = 57,600 inputs
    - FC(57600, 512) -> ReLU -> Dropout(0.3)
    - FC(512, 128) -> ReLU -> Dropout(0.3)
    - FC(128, 3)  # [t_number, wind, pressure]
    """

    def __init__(self):
        super(TyphoonMLP, self).__init__()
        # Calculate flattened input size: 240 * 240 = 57,600
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Linear(57600, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        # Second fully connected layer
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # Output layer
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class TyphoonCNN(nn.Module):
    """Convolutional neural network for typhoon intensity estimation.

    Architecture:
    - Conv2d(1, 16, 3, padding=1) -> BN -> ReLU -> MaxPool(2)    # -> 120x120
    - Conv2d(16, 32, 3, padding=1) -> BN -> ReLU -> MaxPool(2)   # -> 60x60
    - Conv2d(32, 64, 3, padding=1) -> BN -> ReLU -> MaxPool(2)   # -> 30x30
    - Conv2d(64, 128, 3, padding=1) -> BN -> ReLU -> MaxPool(2)  # -> 15x15
    - AdaptiveAvgPool2d(1) -> Flatten
    - FC(128, 64) -> ReLU -> Dropout(0.3)
    - FC(64, 3)   # [t_number, wind, pressure]
    """

    def __init__(self):
        super(TyphoonCNN, self).__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)

        # Adaptive pooling to get fixed size feature vector
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Global pooling and classification
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TyphoonCNNv2(nn.Module):
    """CNN predicting T-number only (single output).

    Same conv backbone as TyphoonCNN, but fc2 outputs 1 value instead of 3.
    Wind/pressure are derived post-hoc via Dvorak lookup in evaluate.py.
    """

    def __init__(self):
        super(TyphoonCNNv2, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(self.adaptive_pool(x))
        x = self.dropout(self.relu5(self.fc1(x)))
        x = self.fc2(x)
        return x