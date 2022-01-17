from os import listdir
from os.path import isfile, join
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import os.path as osp
import os
import time
import torch
import torch.nn.functional as F
from alive_progress import alive_bar
import torch_geometric.transforms as T
import glob
import stanza
from sklearn import metrics
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset, download_url,InMemoryDataset, Data
from stanza.utils.conll import CoNLL


# Data Class for the UD dataset
class UDDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        r = self.root + "/raw"
        onlyfiles = [f for f in listdir(r) if isfile(join(r, f))]
        return onlyfiles

    @property
    def processed_file_names(self):
        r = self.root + "/processed/data_*.pt"
        onlyfiles = glob.glob(r)
        return onlyfiles

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        i = 0
        d = 0
        pos = []
        for raw_path in self.raw_paths:
            d += 1
            # Read data from `raw_path`.
            doc = CoNLL.conll2doc(raw_path)

            n = os.path.splitext(os.path.basename(raw_path))
            n = n[0].split("-")
            label = n[0]
            labels = ["fro_srcmf", "la_proiel", "fr_gsd"]

            print(
                f"Dataset: {d}/{len(self.raw_paths)}  Name : {label}   -> length :{len(doc.sentences)} sentences",
                sep="\n",
            )
            with alive_bar(len(doc.sentences)) as bar:
                for s in doc.sentences:
                    l1 = []
                    l2 = []
                    dist = []
                    pos_sentence = []
                    for el in s.words:
                        # print(f"id: {el.id}  -> Root : {el.head}", sep='\n')
                        if el.head == 0:
                            dist.append(0)
                        else:
                            l1.append(el.id - 1)
                            l2.append(el.head - 1)
                            dist.append(abs(el.head - el.id))

                    # embedding
                    edge_index = torch.tensor([l1, l2])
                    features = [[x] for x in range(1, len(s.words) + 1)]
                    dist = [[el] for el in dist]
                    f = [features[i] + dist[i]
                         for i in range(0, len(features))]
                    x = torch.tensor(f)
                    data = Data(x=x, edge_index=edge_index)

                    # labels
                    labels = ["fro_srcmf", "la_proiel", "fr_gsd"]
                    data.y = torch.tensor(labels.index(label))

                    if self.transform is not None:
                        data = self.transform(data)
                        print(
                            f"Has isolated nodes: {graph.has_isolated_nodes()}")

                    torch.save(
                        data,
                        osp.join(
                            self.processed_dir,
                            "data_{}.pt".format(i)))
                    i += 1

                    bar()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(
                self.processed_dir,
                "data_{}.pt".format(idx)))
        return data

# Data Class for the SemEval dataset


class BenchmarkDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        r = self.root + "/raw"
        onlyfiles = [f for f in listdir(r) if isfile(join(r, f))]
        return onlyfiles

    @property
    def processed_file_names(self):
        r = self.root + "/processed/data_*.pt"
        onlyfiles = glob.glob(r)
        return onlyfiles

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        i = 0
        d = 0
        pos = []
        nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,mwt,pos,lemma,depparse',
            verbose='False')
        for raw_path in self.raw_paths:
            d += 1
            # Read data from `raw_path`.

            labels = ['1650','1700','1750','1800','1850','1900','1950','2000']
            with open(raw_path, "r", encoding="utf-8") as infile:
                sentences = infile.readlines()
                print(
                    f"Dataset: {d}/{len(self.raw_paths)}  Name : {raw_path}   -> length :{len(sentences)} sentences",
                    sep="\n",
                )
                with alive_bar(len(sentences)) as bar:
                    for line in sentences:
                        line = line.split('\t')
                        l1 = []
                        l2 = []
                        dist = []
                        pos_sentence = []
                        s = nlp(line[0])
                        s = s.sentences[0]
                        for el in s.words:
                            #print(f"id: {el.id}  -> Root : {el.head}", sep='\n')
                            if el.head == 0:
                                dist.append(0)
                            else:
                                l1.append(el.id - 1)
                                l2.append(el.head - 1)
                                dist.append(abs(el.head - el.id))

                        # embedding
                        edge_index = torch.tensor([l1, l2])
                        features = [[x] for x in range(1, len(s.words) + 1)]
                        dist = [[el] for el in dist]
                        f = [features[i] + dist[i]
                             for i in range(0, len(features))]
                        x = torch.tensor(f)
                        data = Data(x=x, edge_index=edge_index)

                        # labels
                        label = line[1].strip()
                        data.y = torch.tensor(labels.index(label))
                        if self.transform is not None:
                            data = self.transform(data)
                            print(
                                f"Has isolated nodes: {graph.has_isolated_nodes()}")

                        torch.save(
                            data,
                            osp.join(
                                self.processed_dir,
                                "data_{}.pt".format(i)))
                        i += 1
                        bar()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(
                self.processed_dir,
                "data_{}.pt".format(idx)))
        return data
