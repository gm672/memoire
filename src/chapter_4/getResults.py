from train_neural_network import trainModel, createDatasets
import argparse
import textwrap
from comet_ml import Optimizer
import sys
from torch_geometric.loader import DataLoader
import torch
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import numpy as np
from sklearn.dummy import DummyClassifier

def predict(data_test,model,labelsdict):
    true = []
    predicted = []
    correct = 0
    for data in data_test:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float(), data.edge_index.long(), data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

        true.append([x for x in data.y.cpu().detach().numpy()])
        predicted.append([x for x in pred.cpu().detach().numpy()])
    acc = correct / len(data_test.dataset)

    true = itertools.chain(*true)
    predicted = itertools.chain(*predicted)
    t = []
    p = []
    for el in list(true):
        t.append(labelsdict[el])
    for el in list(predicted):
        p.append(labelsdict[el])

    from sklearn.metrics import f1_score

    # Score
    print(classification_report(t,p))


    # Random Baseline
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(t,t)
    a = dummy_clf.predict(t)
    print('Random Baseline')
    print(classification_report(t,a,zero_division=0))

    if args.cm == 'yes':
        cm = confusion_matrix(y_true=t, y_pred=p, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
        disp.plot()
        plt.show()


parser = argparse.ArgumentParser(description='Get the prediction for a dataset',formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('--dataset',
                    help= textwrap.dedent('''\
                        The dataset to use : UD or SemEval
                        '''))
parser.add_argument('--cm',
                    help= textwrap.dedent('''\
                        Display a confusion matrix for the results of the classification
                        '''))



args = parser.parse_args()
print(args.dataset)

if args.dataset == 'UD':
    data_test= torch.load('../models/data_ud.pth')
    labelsdict ={}
    labelsdict['UDDataset'] = ['latin','afr','frm']
    model = torch.load('../models/model_ud.pt')
    model.eval()
    labels = ['Latin', 'Old French', 'Modern French']
    predict(data_test,model,labels)
elif args.dataset == "SemEval":
    data_test= torch.load('../models/data_semeval.pth')
    labelsdict ={}
    labelsdict['semEval'] = ['1650','1700', '1750', '1800', '1850', '1900','1950','2000']
    model = torch.load('../models/model_semeval.pt')
    model.eval()
    labels = ['1650','1700', '1750', '1800', '1850', '1900','1950','2000']
    predict(data_test,model,labels)
else : 
    print("Error: No dataset of that name")
