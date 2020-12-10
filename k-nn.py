import os

import numpy as np
import h5py
from Bio import SeqIO
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

annotations_path = 'data/annotations/merged_anno.txt'
embeddings_path = 'data/ec_vs_NOec_pide100_c50.h5'
val_data_fasta = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_val.fasta'
test_data_fasta = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_test.fasta'
temp_path = 'data/tmp/'

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

is_cached = os.path.exists(os.path.join(temp_path, 'embeddings.npz'))
if is_cached:
    embeddings = np.load((os.path.join(temp_path, 'embeddings.npz')))
    train_embeddings = embeddings['train']
    val_embeddings = embeddings['val']
    test_embeddings = embeddings['test']

with h5py.File(embeddings_path, 'r') as h5:
    available_identifiers = h5.keys()
    for i, annotation in tqdm(enumerate(open(annotations_path))):
        annotation_array = annotation.strip().split('\t')
        identifier = annotation_array[0]
        # ec number is an array with 4 entries. One entry for each component of the ec number
        ec_number = annotation_array[1].split(';')[0].split('.')
        if identifier in available_identifiers:
            if identifier in val_identifiers:
                if not is_cached:
                    val_embeddings.append(h5[identifier][:])
                val_labels.append(ec_number)
            elif identifier in test_identifiers:
                if not is_cached:
                    test_embeddings.append(h5[identifier][:])
                test_labels.append(ec_number)
            else:
                train_identifiers.append(identifier)
                if not is_cached:
                    train_embeddings.append(h5[identifier][:])
                train_labels.append(ec_number)
if not is_cached:
    np.savez(os.path.join(temp_path, 'embeddings.npz'), train=train_embeddings, val=val_embeddings,
             test=test_embeddings)

print(len(val_labels))
print(len(test_labels))
print(len(train_identifiers))

train_label1 = np.array(train_labels)[:, 0]  # take the first column of the array to get the first ec number
val_label1 = np.array(val_labels)[:, 0]  # take the first column of the array to get the first ec number
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(train_embeddings, train_label1)

pred_val = classifier.predict(val_embeddings)
accuracy = (pred_val == val_label1).sum() / len(val_label1)
predictions_truth = np.stack([pred_val, val_label1])
print(predictions_truth)
print(accuracy)
