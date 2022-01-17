# Memoire
------------
## Description 
The code used in my dissertation.

------------
## Installation
To install all the dependencies needed, use

> pip -r requirements.txt

Note that you can't intall some of pytorch dependenccies in a single requirements.txt (https://github.com/pyg-team/pytorch_geometric/issues/861). First install requirements.txt and then install the last dependencies manually

> torch-spline-conv==1.2.1

> torch-cluster==1.5.9

> torch-scatter==2.0.9

> torch-sparse==0.6.12

------------
## Project Organization


    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── src                <- Source code for use in this project.
    │   ├── chapter_3      <- All the scripts used in chapter 3
    │   │
    │   ├── chapter_4      <- Scripts used in chapter 4
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │                         predictions
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
