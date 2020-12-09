import numpy as np
import h5py
from Bio import SeqIO
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

annotations_path = 'data/annotations/merged_anno.txt'
embeddings_path = 'data/ec_vs_NOec_pide100_c50.h5'
val_data_fasta = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_val.fasta'
test_data_fasta = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_test.fasta'

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

lookup_identifiers = []
lookup_embeddings = []
lookup_labels = []

with h5py.File(embeddings_path, 'r') as h5:
    for annotation in tqdm(open(annotations_path)):
        annotation_array = annotation.strip().split('\t')
        identifier = annotation_array[0]
        # ec number is an array with 4 entries. One entry for each component of the ec number
        ec_number = annotation_array[1].split(';')[0].split('.')

        if identifier in val_identifiers:
            val_identifiers.append(identifier)
            val_embeddings.append(h5[identifier][:])
        elif identifier in test_identifiers:
            test_identifiers.append(identifier)
            test_embeddings.append(h5[identifier][:])
        else:
            lookup_identifiers.append(identifier)
            lookup_embeddings.append(h5[identifier][:])
            lookup_labels.append(ec_number)

lookup_label1 = np.array(lookup_labels)[:, 0]  # take the first column of the array to get the first ec number
val_label1 = np.array(val_labels)[:, 0]  # take the first column of the array to get the first ec number
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(lookup_embeddings, lookup_label1)

pred_val = classifier.predict(val_embeddings)

accuracy = (pred_val == val_label1).sum() / len(val_label1)
print(accuracy)