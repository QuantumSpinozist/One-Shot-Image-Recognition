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





from src.model import SiameseNetwork

from src.dataset import OmniglotDataset, OmniglotDatasetOneShot

from src.train_test import train, test, create_N_way_plots




def main():
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for testing (default: 2048)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--step-size', type=float, default=50, 
                        help='Number of epochs after which lr is decreased (default: 50)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-interval', type=int, default=50,
                        help='number of epochs after which model is saved (default: 50)')
    
    parser.add_argument('--train-model', action='store_true', default=True,
                        help='train the model (set to false with --no-train-model)')
    parser.add_argument('--no-train-model', dest='train_model', action='store_false',
                        help='do not train the model')

    parser.add_argument('--load-last-model', action='store_true', default=True,
                        help='load the last saved state of the model (default: True)')
    parser.add_argument('--no-load-last-model', dest='load_last_model', action='store_false',
                        help='initialize model from scratch')

    parser.add_argument('--n-way-one-shot', action='store_true', default=True,
                        help='evaluate on one-shot tasks for different numbers of classes (default: True)')
    parser.add_argument('--no-n-way-one-shot', dest='n_way_one_shot', action='store_false',
                        help='disable evaluation on one-shot tasks')

    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 7,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = OmniglotDataset('../data', train=True, download=True)
    test_dataset = OmniglotDataset('../data', train=False, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)

    if args.load_last_model:
        model.load_state_dict(torch.load("siamese_network.pt", weights_only=True, map_location=device))

    if args.train_model:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, )
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

            if (epoch % args.save_model_interval == 0) and args.save_model:
                torch.save(model.state_dict(), "siamese_network.pt")
                print(f"Model saved at epoch {epoch}")
    
    if args.save_model:
        torch.save(model.state_dict(), "siamese_network.pt")
    
    
    model.load_state_dict(torch.load("siamese_network.pt", weights_only=True, map_location=device))
    model.eval()
 
    os_dataset = OmniglotDatasetOneShot('../data', train=False, download=False)
    
    
    evaluation_alphabets = ["Angelic", "Atemayar_Qelisayer", "Atlantean", "Aurek-Besh", "Avesta",
                           "Ge_ez", "Glagolitic", "Gurmukhi", "Kannada", "Keble", "Malayalam",
                            "Manipuri", "Mongolian", "Old_Church_Slavonic_(Cyrillic)", "Oriya",
                            "Sylheti", "Syriac_(Serto)", "Tengwar", "Tibetan"]
    
    if args.n_way_one_shot:
        create_N_way_plots(args, model, device, os_dataset, evaluation_alphabets, test_kwargs)
    
    

if __name__ == '__main__': 
    main()
