from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR


import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from src.train_test import one_shot_pred, one_shot_test, train, test, create_N_way_plots

# Dummy model class for testing purposes
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(28*28, 1)  # Simplified for testing purposes

    def forward(self, x1, x2):
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        return torch.sigmoid(self.fc(x1 - x2))

# Fixtures for common setup
@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def model(device):
    return DummyModel().to(device)

@pytest.fixture
def args():
    class Args:
        log_interval = 1
        dry_run = True
    return Args()

@pytest.fixture
def data_loader():
    images_1 = torch.rand(10, 1, 28, 28)
    images_2 = torch.rand(10, 1, 28, 28)
    targets = torch.randint(0, 2, (10,)).float()
    dataset = TensorDataset(images_1, images_2, targets)
    return DataLoader(dataset, batch_size=2)

@pytest.fixture
def one_shot_data_loader():
    images_1 = torch.rand(10, 1, 28, 28)
    targets = torch.randint(0, 2, (10,)).float()
    dataset = TensorDataset(images_1, targets)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def test_loader():
    images_1 = torch.rand(10, 1, 28, 28)
    images_2 = torch.rand(10, 1, 28, 28)
    targets = torch.randint(0, 2, (10,)).float()
    dataset = TensorDataset(images_1, images_2, targets)
    return DataLoader(dataset, batch_size=2)


# Testing the train function
def test_train(args, model, device, data_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train(args, model, device, data_loader, optimizer, epoch=1)
    for param in model.parameters():
        assert param.grad is not None, "Parameters should have gradients after training step."

# Testing the test function
def test_test(model, device, data_loader):
    test(model, device, data_loader)

# Testing one_shot_pred function
def test_one_shot_pred(model, device):
    images = torch.rand(5, 1, 28, 28)
    class_examples = [(torch.rand(1, 28, 28), torch.tensor(i)) for i in range(5)]
    outputs_values, outputs_indices = one_shot_pred(model, images, class_examples, device)
    assert outputs_values.shape == (5,), "Output values should have the shape of the batch size."
    assert outputs_indices.shape == (5,), "Output indices should have the shape of the batch size."

# Testing one_shot_test function
def test_one_shot_test(args, model, device, one_shot_data_loader):
    class_examples = [(torch.rand(1, 28, 28), torch.tensor(i)) for i in range(5)]
    accuracy = one_shot_test(args, model, device, one_shot_data_loader, class_examples)
    assert 0 <= accuracy <= 100, "One-shot test accuracy should be within [0, 100]"


