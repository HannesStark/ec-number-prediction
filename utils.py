from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

ECNUM = ['Oxidoreductases', 'Transferases', 'Hydrolases', 'Lyases', 'Isomerases', 'Ligases', 'Translocase']


def plot_class_accuracies(accuracy, stderr):
    """
    Create seaborn plot
    Args:
        accuracy: accuracies
        stderr: standard errors

    Returns:

    """
    df = pd.DataFrame({'Localization': ECNUM,
                       "Accuracy": accuracy,
                       "std": stderr})
    sn.set_style('darkgrid')
    barplot = sn.barplot(x="Accuracy", y="Localization", data=df, ci=None)
    barplot.set(xlabel='Average accuracy', ylabel='')
    barplot.axvline(1)
    plt.errorbar(x=df['Accuracy'], y=ECNUM, xerr=df['std'], fmt='none', c='black', capsize=3)
    plt.tight_layout()
    plt.show()


def plot_confusion(results: np.ndarray):
    """
    Turns results into two confusion matrices, plots them side by side and writes them to tensorboard
    Args:
        results: [n_samples, 2] the first column is the prediction the second is the true label

    Returns:

    """

    confusion = confusion_matrix(results[:, 1], results[:, 0])  # confusion matrix for train
    confusion = confusion/confusion.sum(axis=0)
    train_cm = pd.DataFrame(confusion)

    sn.heatmap(train_cm, annot=True, cmap='Blues', fmt='.1%', rasterized=False)
    plt.show()

import os
import json
import sys
from Bio import SeqIO
import h5py
import time
def list_itterator(l,load_fun):
    for file in l:
        arr,lenght = load_fun(file)
        for i in range(lenght):
            yield arr[i]
def storeNPZ(path,data,names,other):
    np.savez(path, data=data,names=names,other=other)
def loadNPZ(path):
    a = np.load(path)
    return a['data'],a['names'],a['other']

#First time call: Loads the txt/fasta file, dumpes it in a .json. Returns dic(slow)
#Other times: Load .json. Returns dic. (much faster)
def loadBuffered(path : str):
    t = time.time()
    if os.path.exists(path+'.json'):
        with open(path+'.json') as f:
            a = json.load(f)
            print('Time loadBuffered',path,time.time()-t)
            return a
            
    else:
        dic_id = {}        
        i = 0
        print('First time call. This file is loaded and dupped in to a json. Next time this will not take that much time.')
        if path.endswith('.txt'):
            with open(path) as fp:
                for line in fp:
                    input = line.strip().split('\t')
                    dic_id[input[0]]=input[1]
                    sys.stdout.write('\r' + str(i))
                    i+=1
        elif path.endswith('.fasta'):
            for sequence in SeqIO.parse(path, "fasta"):
                #print(sequence)
                dic_id[sequence.id] = str(sequence.seq)
                sys.stdout.write('\r' + str(i))
                i+=1
        else:
            raise NotImplementedError()
        with open(path +'.json', 'w') as f:
            json.dump(dic_id, f)
        print('Time loadBuffered',path,time.time()-t)
        return dic_id

def join_h5(h5py_file,otherFile,limit=-1):
    t = time.time()
    if otherFile == None:
        with h5py.File(h5py_file, 'r') as h5:
            print('Loading keys, this will really take a while. (On my PC it took 500 s... but don\'t worry. The next part takes even longer (~30 min).)')
            keys = list(h5.keys())
            print('Loaded all keys')
        dic = None
    elif type(otherFile) is str:
        dic = loadBuffered(otherFile)
        keys = dic.keys()
    elif type(otherFile) is list:
        keys = otherFile
        dic = None
    elif type(otherFile) is dict:
        keys = otherFile.keys()
        dic = otherFile
    else:
        raise NotImplementedError()
    identifiers = []
    embeding = []
    otherlist = []
    with h5py.File(h5py_file, 'r') as h5:
        i = 0
        j = 0
        for key in keys:
            if key in h5:
                identifiers.append(key)
                embeding.append(h5[key][:])
                if dic != None:
                    otherlist.append(dic[key])
                i += 1
            else:
                j+=1
                pass
            if i == limit:
                print('\nAfter', i, 'steppt the program stopped adding.')
                break
            sys.stdout.write('\r' + str(i))
        print('\n',i, 'Embedings Found')
        print(j, 'Embendings not Found')
        print('Time join_h5',time.time()-t)
        return identifiers,embeding,otherlist

def join_h5_buffer(h5py_file,otherFile,limit=-1,store=False):
    t = time.time()
    if otherFile == None:
        if limit == -1:
            buffername = h5py_file+'.npz'
        else:
            buffername = h5py_file+'_'+str(limit)+'.npz'
    else:
        if limit == -1:
            buffername = otherFile+'.npz'
        else:
            buffername = otherFile+'_'+str(limit)+'.npz'
    if os.path.exists(buffername):
        a =  loadNPZ(buffername)
        print('Time join_h5_buffer',time.time()-t)
        return a
    else:
        if store:
            a,b,c = join_h5(h5py_file,otherFile,limit=limit)
            storeNPZ(buffername,a,b,c,)
            print('Time join_h5_buffer',time.time()-t)
            return a,b,c
        else:
            a=join_h5(h5py_file,otherFile,limit=limit)
            print('Time join_h5_buffer',time.time()-t)
            return a

def get_h5():
    return join_h5_buffer('data/ec_vs_NOec_pide100_c50.h5',None,store=True)[:-1]

#computes the confusionsmatrix an plots it.
def plot_multiclass_all(val_label1,pred_val,label=ECNUM):
    from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
    f1 = f1_score(val_label1, pred_val, average='macro')
    print('F1 :', f1)

    mccs = []
    accuracies = []
    class_accuracies = []
    results = np.stack([pred_val, val_label1])
    
    accuracies.append(100 * (results[0] == results[1]).sum() / len(results[0]))
    mccs.append(matthews_corrcoef(results[1], results[0]))
    conf = confusion_matrix(results[1], results[0])
    class_accuracies.append(np.diag(conf) / conf.sum(1))

    accuracy = np.mean(accuracies)
    accuracy_stderr = np.std(accuracies)
    mcc = np.mean(mccs)
    mcc_stderr = np.std(mccs)
    class_accuracy = np.mean(np.array(class_accuracies), axis=0)
    class_accuracy_stderr = np.std(np.array(class_accuracies), axis=0)
    results_string = 'Accuracy: {:.2f}% \n' \
                    'Accuracy stderr: {:.2f}%\n' \
                    'MCC: {:.4f}\n' \
                    'MCC stderr: {:.4f}\n'.format(accuracy, accuracy_stderr, mcc, mcc_stderr)
    print(results_string)
    if len(class_accuracy) == len(label):
        plot_class_accuracies(class_accuracy, class_accuracy_stderr)
    plot_confusion(results.T)