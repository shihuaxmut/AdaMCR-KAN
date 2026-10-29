
import random

random.seed(1234)

import warnings

warnings.filterwarnings("ignore")
import os
import torch

import torch.utils.data as Data

os.environ['CUDA_VISIBLE_DEVICES'] = '6'



def make_pMHCI_data(data,vocab, pep_max_len=25,hla_max_len=34):
    data['peptide'] = data['peptide'].fillna('').astype(str)
    data['HLA_sequence'] = data['HLA_sequence'].fillna('').astype(str)
    pep_inputs, hla_inputs, labels = [], [], []
    for pep, hla, label in zip(data.peptide, data.HLA_sequence, data.label):
        pep, hla = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]
        hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)


class pMHCI_dataset(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, labels):
        super(pMHCI_dataset, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]


def pMHCI_data_with_loader(data, vocab, batch_size=1024,pep_max_len=25,hla_max_len=34,shuffle=True):
    pep_inputs, hla_inputs, labels = make_pMHCI_data(data,vocab,pep_max_len,hla_max_len)
    loader = Data.DataLoader(pMHCI_dataset(pep_inputs, hla_inputs, labels), batch_size, shuffle,num_workers=0)
    return pep_inputs, hla_inputs, labels, loader


def make_pMHCI_TCR_data(data,vocab, pep_max_len=25,hla_max_len=34, tcr_max_len=27):
    data['tcr'] = data['tcr'].fillna('').astype(str)
    data['peptide'] = data['peptide'].fillna('').astype(str)
    data['HLA_sequence'] = data['HLA_sequence'].fillna('').astype(str)
    tcr_inputs, pep_inputs, hla_inputs, labels = [], [], [], []
    for tcr, pep, hla, label in zip(data.tcr, data.peptide, data.HLA_sequence, data.label):
        tcr, pep, hla = tcr.ljust(tcr_max_len, '-'), pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
        tcr_input = [[vocab[n] for n in tcr]]
        pep_input = [[vocab[n] for n in pep]]
        hla_input = [[vocab[n] for n in hla]]
        tcr_inputs.extend(tcr_input)
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        labels.append(label)
    return torch.LongTensor(tcr_inputs), torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(
        labels)


class pMHCI_TCR_train_dataset(Data.Dataset):
    def __init__(self, postcr_inputs, pep_inputs, hla_inputs, negtcr_inputs, labels):
        super(pMHCI_TCR_train_dataset, self).__init__()
        self.postcr_inputs = postcr_inputs
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.negtcr_inputs = negtcr_inputs
        self.labels = labels

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.postcr_inputs[idx], self.pep_inputs[idx], self.hla_inputs[idx], self.negtcr_inputs[idx], self.labels[idx]


class pMHCI_TCR_test_dataset(Data.Dataset):
    def __init__(self, tcr_inputs, pep_inputs, hla_inputs, labels):
        super(pMHCI_TCR_test_dataset, self).__init__()
        self.tcr_inputs = tcr_inputs
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels

    def __len__(self):  # 样本数
        return self.pep_inputs.shape[0]  # 改成hla_inputs也可以哦！

    def __getitem__(self, idx):
        return self.tcr_inputs[idx], self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]


def pMHCI_TCR_train_data_with_loader(posdata, negdata,vocab, batch_size=1024, pep_max_len=20,hla_max_len=34,
                                     tcr_max_len=27,shuffle=True):
    postcr_inputs, pep_inputs, hla_inputs, labels = make_pMHCI_TCR_data(posdata,vocab,pep_max_len,hla_max_len,
                                     tcr_max_len)
    negtcr_inputs, pep_inputs_neg, hla_inputs_neg, labels_neg = make_pMHCI_TCR_data(negdata,vocab,pep_max_len,hla_max_len,
                                     tcr_max_len)
    loader = Data.DataLoader(pMHCI_TCR_train_dataset(postcr_inputs, pep_inputs, hla_inputs, negtcr_inputs, labels), batch_size,
                             shuffle, num_workers=0)
    return postcr_inputs, pep_inputs, hla_inputs, negtcr_inputs, labels, loader


def pMHCI_TCR_test_data_with_loader(data, vocab, batch_size=1024,pep_max_len=20,hla_max_len=34,
                                     tcr_max_len=27, shuffle=False):
    tcr_inputs, pep_inputs, hla_inputs, labels = make_pMHCI_TCR_data(data,vocab,pep_max_len,hla_max_len,
                                     tcr_max_len)
    loader = Data.DataLoader(pMHCI_TCR_test_dataset(tcr_inputs, pep_inputs, hla_inputs, labels), batch_size, shuffle,
                             num_workers=0)
    return tcr_inputs, pep_inputs, hla_inputs, labels, loader




