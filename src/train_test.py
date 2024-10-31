from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Trains the model for one epoch.

    Parameters:
    - args: Command line arguments containing configurations.
    - model: The neural network model to train.
    - device: The device to train on (CPU or GPU).
    - train_loader: DataLoader for the training dataset.
    - optimizer: The optimizer used for training.
    - epoch: Current epoch number (for logging).
    """
    model.train()
    criterion = nn.BCELoss()  # Using Binary Cross Entropy Loss

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    """
    Evaluates the model on the test dataset.

    Parameters:
    - model: The neural network model to evaluate.
    - device: The device to evaluate on (CPU or GPU).
    - test_loader: DataLoader for the test dataset.

    Returns:
    - None: Prints the average loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.BCELoss()  # Using Binary Cross Entropy Loss

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # Sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # Get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def one_shot_pred(model, images, class_examples, device):
    """
    Performs one-shot prediction using the model.

    Parameters:
    - model: The neural network model to use for predictions.
    - images: Tensor containing the images to classify.
    - class_examples: List of class examples for one-shot learning.
    - device: The device to perform calculations on (CPU or GPU).

    Returns:
    - outputs_values: Tensor containing the predicted values.
    - outputs_indices: Tensor containing the predicted class indices.
    """
    number_of_images = images.shape[0]
    outputs_list = []
    for i, class_example in enumerate(class_examples):
        class_example = class_example[0].unsqueeze(0).repeat(number_of_images, 1, 1, 1).to(device)
        outputs = model(images.to(device), class_example).squeeze()
        outputs_list.append(outputs)

    outputs_stacked = torch.stack(outputs_list)
    outputs_values, outputs_indices = torch.max(outputs_stacked, dim=0)

    return outputs_values, outputs_indices + class_examples[0][1]


def one_shot_test(args, model, device, test_loader, class_examples):
    """
    Tests the model using one-shot learning.

    Parameters:
    - args: Command line arguments containing configurations.
    - model: The neural network model to test.
    - device: The device to test on (CPU or GPU).
    - test_loader: DataLoader for the test dataset.
    - class_examples: List of class examples for one-shot learning.

    Returns:
    - float: The total accuracy of the model.
    """
    model.eval()
    correct = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            outputs_values, outputs_indices = one_shot_pred(model, images, class_examples, device)

            hits = torch.sum(outputs_indices == targets).item()
            correct += hits
            
            if batch_idx % args.log_interval == 0:
                print(f'Progress {100. * batch_idx / len(test_loader):.2f}')
                print(f'Accuracy {100 * hits / len(images):.2f}')

    total_accuracy = 100 * correct / len(test_loader.dataset)
    print('Total accuracy: ' + str(total_accuracy))
    return total_accuracy


def create_N_way_plots(args, model, device, os_dataset, evaluation_alphabets, test_kwargs, N_list=[2, 5, 10, 15, 20]):
    """
    Creates plots for N-way one-shot learning accuracies.

    Parameters:
    - args: Command line arguments containing configurations.
    - model: The neural network model to use for predictions.
    - device: The device to perform calculations on (CPU or GPU).
    - os_dataset: The one-shot dataset to evaluate.
    - evaluation_alphabets: List of alphabets for evaluation.
    - test_kwargs: Additional keyword arguments for DataLoader.
    - N_list: List of N values for N-way classification.

    Returns:
    - None: Generates and saves plots.
    """
    total_accuracies = []
    for N in N_list:
        total_acc = 0
        alphabet_accuracies = {}

        for alphabet in evaluation_alphabets:
            os_dataset.update_alphabet(alphabet=alphabet, N=N)
            os_test_loader = torch.utils.data.DataLoader(os_dataset, **test_kwargs)

            acc = one_shot_test(args, model, device, os_test_loader, os_dataset.class_examples)
            alphabet_accuracies[alphabet] = acc
            total_acc += acc

        # Calculate and store the overall accuracy
        overall_accuracy = total_acc / len(evaluation_alphabets)
        total_accuracies.append(overall_accuracy)

        # Plot bar plot for each N
        plt.figure(figsize=(8, 5))
        bars = plt.bar(alphabet_accuracies.keys(), alphabet_accuracies.values(), color='skyblue')

        plt.xlabel("Alphabets")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{N}-way One-Shot Accuracies by Alphabet")
        plt.xticks(rotation=90)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.0f}", ha="center", va="bottom")

        plt.savefig(f"{N}_way_one_shot_barplot.png", format="png", bbox_inches="tight")
        plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(N_list, total_accuracies, label="Mean Accuracy", marker='o', color='b')

    # Labels and title
    plt.xlabel("N (way)")
    plt.ylabel("Accuracy (%)")
    plt.title("One-Shot Accuracy by N-way Classification")
    plt.legend()

    # Save the final plot
    plt.savefig('N_way_one_shot_accuracy_comparison.png', format='png', bbox_inches="tight")
    plt.show()
