# Chapter 4

To train a neural network, use the file optimize.py. the script records a series of experiments inside the Comet-ML dashboard. To use this file, you need a Comet-ML api key inside a `.comet.config` file. Registering is free using a educational email adress. Inside the script, you can change the parameters of the experiments.


Example of usage :
`python run_experiment.py --training_size=6000 --test_size=600 --batch_size=64 --epochs=500 UDDataset`


To predict result using a model, run the predict.py script. By default, the script uses the test set and the best model for each dataset.

Example of usage :
`python getResults.py --dataset=UD --cm=no `
