from train_neural_network import trainModel, createDatasets
import argparse
import textwrap
from comet_ml import Optimizer
import sys
from torch_geometric.loader import DataLoader


parser = argparse.ArgumentParser(description='Train a neural network.',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('data', default='UDDataset',
                    help= textwrap.dedent('''\
                        Choses the data used for the training :
                        - UDDataset : 3 languages (latin, old french, modern french)
                        - semEval : 8 intervals of 50 years of the english languages between 1650-2000

                        '''))

parser.add_argument('--training_size', default='1000',
                    help= textwrap.dedent('''\
                        The number of training samples.
                        They will be balanced accross target classes.

                        '''))

parser.add_argument('--test_size', default='100',
                    help= textwrap.dedent('''\
                        The number of test samples.
                        They will be balanced accross target classes.

                        '''))

parser.add_argument('--batch_size', default='64',
                    help= textwrap.dedent('''\
                        Batch size in the data loader.

                        '''))
parser.add_argument('--epochs', default='100',
                    help= textwrap.dedent('''\
                        Number of training epochs.

                        '''))
parser.add_argument('--load_model',
                    help= textwrap.dedent('''\
                        Path of a saved model.
                        #TOFDO
                        
                        '''))


args = parser.parse_args()

#log
import comet_ml
import logging
comet_ml.init()


config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {
        "hidden_channels": {"type": "integer", "min": 12, "max": 72},
        "learning_rate": {"type": "float", "min": 0.0001, "max": 0.005}
    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "test_accuracy",
        "objective": "maximize",
    },
}

opt = comet_ml.Optimizer(config)
data = createDatasets(args)
i = 1
for experiment in opt.get_experiments(project_name="learning rate opti"):
    results = trainModel(data[0],data[1],data[2],args,experiment,i)
    i +=1


