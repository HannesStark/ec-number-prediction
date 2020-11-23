from Bio.SeqIO import PirIO
import h5py
from time import time
import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
import seaborn as sns

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-h5", "--h5py_file", type=str,default='data/ec_vs_NOec_pide100_c50.h5', help="Path to data/ec_vs_NOec_pide100_c50.h5")
    parser.add_argument("-anno", "--annotations", type=str,default='data/annotations/merged_anno.txt', help="Path to data/annotations/merged_anno.txt")
    parser.add_argument("-br", "--breaking", type=int,default=1000 , help="Number of random samples after aquesitons stopps. (-1 == inf)")
    parser.add_argument("-pp", "--perplexity", type=int,default=30 , help="perplexity")
    parser.add_argument("-ni", "--n_iter", type=int,default=1000 , help="n_iter")
    #parser.add_argument("-br", "--breaking", type=int,default=-1 , help="Number of random samples after aquesitons stopps. (-1 == inf)")
    args = parser.parse_args()
    print(args)
   
    h5py_file = args.h5py_file
    fasta_path = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_train.fasta'
    anno = args.anno

    print(h5py_file)
    proteins = []
    print('LOAD - This may take a while, if you run this the first time!')
    count = 0

    # First I want to see the splitt only in the 6 main-classes, befor we do something fancy here.
    def reduceAnno(anno: str):
        i = int(anno[0])
        return i


    identifiers = []
    embeding = []
    color = []
    with open(anno) as fp:
        with h5py.File(h5py_file, 'r') as h5:
            i = 0
            j = 0
            for line in fp:
                input = line.strip().split('\t')
                if input[0] in h5:
                    identifiers.append(input[0])
                    embeding.append(h5[input[0]][:])
                    color.append(reduceAnno(input[1]))
                    i += 1
                else:
                    pass
                if i == args.breaking:
                    print('After', i, 'steppt the program stopped adding.')
                    break
                sys.stdout.write('\r' + str(i))

    perplexity = args.perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=args.n_iter, verbose=1)

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", len(np.unique(np.array(color))))
    print('Start transform')
    t0 = time()
    Y = tsne.fit_transform(embeding)
    t1 = time()
    print("%s: %.2g sec" % ('TSNE', t1 - t0))
    sns.scatterplot(Y[:, 0], Y[:, 1], hue=color, legend='full', palette=palette)
    plt.title(str(perplexity))
    plt.show()
    #################################################################################################################
    # Source https://reneshbedre.github.io/blog/tsne.html
    # https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
    
