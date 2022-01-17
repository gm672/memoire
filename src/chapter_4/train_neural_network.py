import itertools
from comet_ml import Experiment
import csv
from sklearn import metrics
import argparse
from data import BenchmarkDataset, UDDataset
from model import GCN
import textwrap
from torch.utils.data import ConcatDataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.loader import DataLoader


def createDatasets(args):
    # Target classes
    labelsdict = {}
    labelsdict['UDDataset'] = ['latin', 'afr', 'frm']
    labelsdict['semEval'] = ['1650','1700','1750','1800','1850','1900','1950','2000']
    dataTraining = {}
    dataTest = {}
    # Number of training samples from each classes
    ntraining = int(int(args.training_size) / len(labelsdict[args.data]))
    # Number of test samples from each classes
    ntest = int(int(args.test_size) / len(labelsdict[args.data]))
    # Creating the datasets
    if args.data == 'semEval' or args.data == 'semEvalCoarse':
        for el in labelsdict[args.data]:
            # training datasets
            path = "../../data/{0}/{1}/training".format(args.data, el)
            dataTraining[el] = BenchmarkDataset(path)
            dataTraining[el] = dataTraining[el].shuffle()
            dataTraining[el] = dataTraining[el][:ntraining]
            dataTraining[el].num_classes = len(labelsdict[args.data])
            # test datasets
            path = "../../data/{0}/{1}/val".format(args.data, el)
            dataTest[el] = BenchmarkDataset(path)
            dataTest[el] = dataTest[el].shuffle()
            dataTest[el] = dataTest[el][:ntest]
            dataTest[el].num_classes = len(labelsdict[args.data])
    elif args.data == 'UDDataset':
        for el in labelsdict[args.data]:
            # training datasets
            path = "../../data/{0}/{1}/training".format(args.data, el)
            dataTraining[el] = UDDataset(path)
            dataTraining[el] = dataTraining[el].shuffle()
            dataTraining[el] = dataTraining[el][:ntraining]
            dataTraining[el].num_classes = len(labelsdict[args.data])
            # test datasets
            path = "../../data/{0}/{1}/val".format(args.data, el)
            dataTest[el] = UDDataset(path)
            dataTest[el] = dataTest[el].shuffle()
            dataTest[el] = dataTest[el][:ntest]
            dataTest[el].num_classes = len(labelsdict[args.data])
    else:
        sys.exit("Error : The data chosen is invalid.")
    # Concat dataset
    training_data = []
    test_data = []
    for el in labelsdict[args.data]:
        for d in dataTraining[el]:
            training_data.append(d)
        for d in dataTest[el]:
            test_data.append(d)
    # Print data
    print(f"Training Data")
    print("====================")
    print(f"Number of graphs: {len(training_data)}")
    print(f"\n")
    print(f"Test Data")
    print("====================")
    print(f"Number of graphs: {len(test_data)}")
    #Put in dataLoader
    train_loader = DataLoader(training_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    # Return
    return [train_loader, test_loader, len(labelsdict[args.data])]


def trainModel(train_loader, test_loader, classes, args, experiment, i):
    model = GCN(hidden_channels=experiment.get_parameter(
        "hidden_channels"), num_node_features=2, num_classes=classes)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment.get_parameter("learning_rate"))
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        for data in train_loader:  
            out = model(
                data.x.float(), data.edge_index.long(), data.batch
            )  # forward pass.
            loss = criterion(out, data.y)  # loss.
            loss.backward()  #  gradients.
            optimizer.step()  # Update parameters 
            optimizer.zero_grad()  # Clear gradients
            experiment.log_metric("loss_epoch", loss, step=epoch) # Log training acc into comet

    def test(loader):
        model.eval()
        true = []
        predicted = []
        correct = 0
        # Iterate in batches over the training/test dataset.
        for data in loader:
            out = model(data.x.float(), data.edge_index.long(), data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            # Check against ground-truth labels.
            correct += int((pred == data.y).sum())
            true.append([x for x in data.y.cpu().detach().numpy()])
            predicted.append([x for x in pred.cpu().detach().numpy()])
        acc = correct / len(loader.dataset)
        return [acc, true, predicted]  # Correct predictions.

    for epoch in range(1, (int(args.epochs) + 1)):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        experiment.log_metric(
            "training_accuracy",
            train_acc[0],
            step=epoch)  # Log training acc into comet
        experiment.log_metric(
            "test_accuracy",
            test_acc[0],
            step=epoch)  # Log test acc into comet
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc[0]:.4f},Test Acc: {test_acc[0]:.4f}"
        )
    # Specify a path
    PATH = "datation" + str(i) + ".pt"

    # Save
    torch.save(model, PATH)
    return [test_acc[1], test_acc[2]]