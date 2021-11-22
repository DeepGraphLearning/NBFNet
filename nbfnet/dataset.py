import os
import csv
import glob
from tqdm import tqdm
from ogb import linkproppred

import torch
from torch.utils import data as torch_data

from torchdrug import data, datasets, utils
from torchdrug.core import Registry as R


class InductiveKnowledgeGraphDataset(data.KnowledgeGraphDataset):

    def load_inductive_tsvs(self, train_files, test_files, verbose=0):
        assert len(train_files) == len(test_files) == 2
        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % txt_file, utils.get_line_count(txt_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token = tokens
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        train_entity_vocab, inv_train_entity_vocab = self._standarize_vocab(None, inv_train_entity_vocab)
        test_entity_vocab, inv_test_entity_vocab = self._standarize_vocab(None, inv_test_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(None, inv_relation_vocab)

        self.train_graph = data.Graph(triplets[:num_samples[0]],
                                      num_node=len(train_entity_vocab), num_relation=len(relation_vocab))
        self.valid_graph = self.train_graph
        self.test_graph = data.Graph(triplets[sum(num_samples[:2]): sum(num_samples[:3])],
                                     num_node=len(test_entity_vocab), num_relation=len(relation_vocab))
        self.graph = self.train_graph
        self.triplets = torch.tensor(triplets[:sum(num_samples[:2])] + triplets[sum(num_samples[:3]):])
        self.num_samples = num_samples[:2] + num_samples[3:]
        self.train_entity_vocab = train_entity_vocab
        self.test_entity_vocab = test_entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_train_entity_vocab = inv_train_entity_vocab
        self.inv_test_entity_vocab = inv_test_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab

    def __getitem__(self, index):
        return self.triplets[index]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.CoraLinkPrediction")
class CoraLinkPrediction(datasets.Cora):

    def __init__(self, **kwargs):
        super(CoraLinkPrediction, self).__init__(**kwargs)
        self.transform = None

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def split(self, ratios=(85, 5, 10)):
        length = self.graph.num_edge
        norm = sum(ratios)
        lengths = [int(r / norm * length) for r in ratios]
        lengths[-1] = length - sum(lengths[:-1])

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, lengths, generator=g)


@R.register("datasets.CiteSeerLinkPrediction")
class CiteSeerLinkPrediction(datasets.CiteSeer):

    def __init__(self, **kwargs):
        super(CiteSeerLinkPrediction, self).__init__(**kwargs)
        self.transform = None

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def split(self, ratios=(85, 5, 10)):
        length = self.graph.num_edge
        norm = sum(ratios)
        lengths = [int(r / norm * length) for r in ratios]
        lengths[-1] = length - sum(lengths[:-1])

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, lengths, generator=g)


@R.register("datasets.PubMedLinkPrediction")
class PubMedLinkPrediction(datasets.PubMed):

    def __init__(self, **kwargs):
        super(PubMedLinkPrediction, self).__init__(**kwargs)
        self.transform = None

    def __getitem__(self, index):
        return self.graph.edge_list[index]

    def __len__(self):
        return self.graph.num_edge

    def split(self, ratios=(85, 5, 10)):
        length = self.graph.num_edge
        norm = sum(ratios)
        lengths = [int(r / norm * length) for r in ratios]
        lengths[-1] = length - sum(lengths[:-1])

        g = torch.Generator()
        g.manual_seed(0)
        return torch_data.random_split(self, lengths, generator=g)


@R.register("datasets.FB15k237Inductive")
class FB15k237Inductive(InductiveKnowledgeGraphDataset):

    train_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",
    ]

    test_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_files = []
        for url in self.train_urls:
            url = url % version
            save_file = "fb15k237_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            train_files.append(txt_file)
        test_files = []
        for url in self.test_urls:
            url = url % version
            save_file = "fb15k237_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            test_files.append(txt_file)

        self.load_inductive_tsvs(train_files, test_files, verbose=verbose)


@R.register("datasets.WN18RRInductive")
class WN18RRInductive(InductiveKnowledgeGraphDataset):

    train_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",
    ]

    test_urls = [
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
        "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
    ]

    def __init__(self, path, version="v1", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        train_files = []
        for url in self.train_urls:
            url = url % version
            save_file = "wn18rr_%s_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            train_files.append(txt_file)
        test_files = []
        for url in self.test_urls:
            url = url % version
            save_file = "wn18rr_%s_ind_%s" % (version, os.path.basename(url))
            txt_file = os.path.join(path, save_file)
            if not os.path.exists(txt_file):
                txt_file = utils.download(url, self.path, save_file=save_file)
            test_files.append(txt_file)

        self.load_inductive_tsvs(train_files, test_files, verbose=verbose)


@R.register("datasets.OGBLBioKG")
class OGBLBioKG(data.KnowledgeGraphDataset):

    def __init__(self, path, verbose=1):
        path = os.path.expanduser(path)
        self.path = path

        dataset = linkproppred.LinkPropPredDataset("ogbl-biokg", path)
        self.load_ogb(dataset, verbose=verbose)

    def load_ogb(self, dataset, verbose=1):
        entity_vocab = []
        relation_vocab = []
        entity_type_vocab = []
        inv_entity_type_offset = {}
        entity_type2num = []

        zip_files = glob.glob(os.path.join(dataset.root, "mapping/*.gz"))
        for zip_file in zip_files:
            csv_file = utils.extract(zip_file)
            type = os.path.basename(csv_file).split("_")[0]
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin)
                if verbose:
                    reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
                fields = next(reader)
                if "relidx" in csv_file:
                    for index, token in reader:
                        relation_vocab.append(token)
                else:
                    entity_type_vocab.append(type)
                    inv_entity_type_offset[type] = len(entity_vocab)
                    num_entity = 0
                    for index, token in reader:
                        entity_vocab.append("%s (%s)" % (type, token))
                        num_entity += 1
                    entity_type2num.append(num_entity)

        edge_split = dataset.get_edge_split()
        triplets = []
        num_samples = []
        num_samples_with_neg = []
        negative_heads = []
        negative_tails = []
        for key in ["train", "valid", "test"]:
            split_dict = edge_split[key]
            h = torch.as_tensor(split_dict["head"])
            t = torch.as_tensor(split_dict["tail"])
            r = torch.as_tensor(split_dict["relation"])
            h_type = torch.tensor([inv_entity_type_offset[h] for h in split_dict["head_type"]])
            t_type = torch.tensor([inv_entity_type_offset[t] for t in split_dict["tail_type"]])

            h = h + h_type
            t = t + t_type
            triplet = torch.stack([h, t, r], dim=-1)
            triplets.append(triplet)
            num_samples.append(len(h))
            if "head_neg" in split_dict:
                neg_h = torch.as_tensor(split_dict["head_neg"])
                neg_t = torch.as_tensor(split_dict["tail_neg"])
                neg_h = neg_h + h_type.unsqueeze(-1)
                neg_t = neg_t + t_type.unsqueeze(-1)
                negative_heads.append(neg_h)
                negative_tails.append(neg_t)
                num_samples_with_neg.append(len(h))
            else:
                num_samples_with_neg.append(0)
        triplets = torch.cat(triplets)

        self.load_triplet(triplets, entity_vocab=entity_vocab, relation_vocab=relation_vocab)
        entity_type_vocab, inv_entity_type_vocab = self._standarize_vocab(entity_type_vocab, None)
        self.entity_type_vocab = entity_type_vocab
        self.inv_entity_type_vocab = inv_entity_type_vocab
        self.num_samples = num_samples
        self.num_samples_with_neg = num_samples_with_neg
        self.negative_heads = torch.cat(negative_heads)
        self.negative_tails = torch.cat(negative_tails)

        node_type = []
        for i, num_entity in enumerate(entity_type2num):
            node_type += [i] * num_entity
        with self.graph.node():
            self.graph.node_type = torch.tensor(node_type)

    def split(self, test_negative=True):
        offset = 0
        neg_offset = 0
        splits = []
        for num_sample, num_sample_with_neg in zip(self.num_samples, self.num_samples_with_neg):
            if test_negative and num_sample_with_neg:
                pos_h, pos_t, pos_r = self[offset: offset + num_sample].t()
                neg_h = self.negative_heads[neg_offset: neg_offset + num_sample_with_neg]
                neg_t = self.negative_tails[neg_offset: neg_offset + num_sample_with_neg]
                num_negative = neg_h.shape[-1]
                h = pos_h.unsqueeze(-1).repeat(2, num_negative + 1)
                t = pos_t.unsqueeze(-1).repeat(2, num_negative + 1)
                r = pos_r.unsqueeze(-1).repeat(2, num_negative + 1)
                t[:num_sample_with_neg, 1:] = neg_t
                h[num_sample_with_neg:, 1:] = neg_h
                split = torch.stack([h, t, r], dim=-1)
            else:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
            neg_offset += num_sample_with_neg
        return splits
