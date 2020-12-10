import os

import numpy as np
import h5py
from Bio import SeqIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from tqdm import tqdm

from utils import ECNUM, plot_class_accuracies, plot_confusion

annotations_path = 'data/annotations/merged_anno.txt'
embeddings_path = 'data/ec_vs_NOec_pide100_c50.h5'
val_data_fasta = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_val.fasta'
test_data_fasta = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_test.fasta'
temp_path = 'data/tmp/'
data = 'val'
results_file = data + 'knn.npy'
embeddings_file = data + 'embeddings.npz'

val_identifiers = []
val_embeddings = []
val_labels = []
for sequence in SeqIO.parse(val_data_fasta, "fasta"):
    val_identifiers.append(sequence.id)

test_identifiers = []
test_embeddings = []
test_labels = []
for sequence in SeqIO.parse(test_data_fasta, "fasta"):
    test_identifiers.append(sequence.id)

train_identifiers = []
train_embeddings = []
train_labels = []

if not os.path.exists(temp_path):
    os.mkdir(temp_path)

results_cached = os.path.exists(os.path.join(temp_path, results_file))
embeddings_cached = os.path.exists(os.path.join(temp_path, embeddings_file))
if embeddings_cached:
    embeddings = np.load((os.path.join(temp_path, embeddings_file)))
    train_embeddings = embeddings['train']
    val_embeddings = embeddings['val']
    test_embeddings = embeddings['test']

if not results_cached:
    with h5py.File(embeddings_path, 'r') as h5:
        available_identifiers = h5.keys()
        for i, annotation in tqdm(enumerate(open(annotations_path))):
            annotation_array = annotation.strip().split('\t')
            identifier = annotation_array[0]
            # ec number is an array with 4 entries. One entry for each component of the ec number
            ec_number = annotation_array[1].split(';')[0].split('.')
            if identifier in available_identifiers:
                if identifier in val_identifiers:
                    if not embeddings_cached:
                        val_embeddings.append(h5[identifier][:])
                    val_labels.append(ec_number)
                elif identifier in test_identifiers:
                    if not embeddings_cached:
                        test_embeddings.append(h5[identifier][:])
                    test_labels.append(ec_number)
                else:
                    train_identifiers.append(identifier)
                    if not embeddings_cached:
                        train_embeddings.append(h5[identifier][:])
                    train_labels.append(ec_number)
    train_label1 = np.array(train_labels)[:, 0]  # take the first column of the array to get the first ec number
    val_label1 = np.array(val_labels)[:, 0]  # take the first column of the array to get the first ec number
if not embeddings_cached:
    np.savez(os.path.join(temp_path, embeddings_file), train=train_embeddings, val=val_embeddings,
             test=test_embeddings)

print('number validation points :', len(val_labels))
print('number test points :', len(test_labels))
print('number train points :', len(train_identifiers))

if results_cached:
    results = np.load(os.path.join(temp_path, results_file))
    pred_val = results[0]
    val_label1 = results[1]
else:
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(train_embeddings, train_label1)
    pred_val = classifier.predict(val_embeddings)
results = np.stack([pred_val, val_label1])
np.save(os.path.join(temp_path, results_file), results)

f1 = f1_score(val_label1, pred_val, average='macro')
print('F1 :', f1)

mccs = []
accuracies = []
class_accuracies = []

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

plot_class_accuracies(class_accuracy, class_accuracy_stderr)
plot_confusion(results.T)
