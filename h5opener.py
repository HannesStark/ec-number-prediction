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
    h5py_file= 'data/ec_vs_NOec_pide100_c50.h5'
    fasta_path = 'data/nonRed_dataset/ec_vs_NOec_pide20_c50_train.fasta'
    anno = 'data/annotations/merged_anno.txt'
    
    print(h5py_file)
    proteins = []
    print('LOAD - This may take a while, if you run this the first time!')
    #https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
    count = 0
    #with h5py.File(h5py_file, 'r') as f:
    #    first_key = list(f.keys())[0]
    #    print("The first protein in the set has id {} and the embedding is of size {}.".format(first_key, f[first_key].shape[0]))
    #    print('The size is of the set is',len(f.keys()))   
    #    #    l = len(f.keys())
    #    #    print(list(f['A0A140JWT2']))
    #    #    for protein_id in f.keys():
    #    #        #print('protein_id',protein_id)
    #    #        proteins.append((protein_id, list(f[protein_id])))
    #    #        count+=1
    #    #        sys.stdout.write('\r'+str(count)+"/"+str(l))
    #    #        #if count == 1000:
    #    #        #    print('Stop loading after 1000 protiens')
    #    #        #    break

    #First I want to see the splitt only in the 6 main-classes, befor we do something fancy here.
    def reduceAnno(anno : str):
        i = int(anno[0])
        return i
    identifiers = []
    embeding = []
    color = []
    with open(anno) as fp:
        with h5py.File(h5py_file, 'r') as h5:
            i = 0
            for line in fp:
                input = line.strip().split('\t')
                if input[0] in h5:
                    identifiers.append(input[0])
                    #print(h5[input[0]])
                    embeding.append(h5[input[0]][:])
                    color.append(reduceAnno(input[1]))
                    i +=1
                else:
                    pass
                    #print(input[0], 'NOT FOUND')
                if i == 1000:
                    print('After', i, 'steppt i stop counted.')
                    break
                sys.stdout.write('\r'+str(i))

    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1)
    
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("bright", len(np.unique(np.array(color))))
    print('Start transform')
    t0 = time()
    Y = tsne.fit_transform(embeding)
    t1 = time()
    print("%s: %.2g sec" % ('TSNE', t1 - t0))
    sns.scatterplot(Y[:,0], Y[:,1], hue=color, legend='full', palette=palette)
    plt.show()
    #plt.scatter(Y[:, 0], Y[:, 1], 'o', );
    #ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
    #fig.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    #fig.set_title("%s (%.2g sec)" % ('TSNE', t1 - t0))
    #fig.xaxis.set_major_formatter(NullFormatter())
    #fig.yaxis.set_major_formatter(NullFormatter())
    #fig.axis('tight')
    
    #plt.show()
    #################################################################################################################
    ##Source https://reneshbedre.github.io/blog/tsne.html
    #df = pd.DataFrame(list(zip(identifiers,embeding,color)),
    #              columns=['identifier', 'embeding','class'])
    #data = get_data('embeding').data
    #print(df.head())
    #from sklearn.manifold import TSNE
    #tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=10, verbose=1).fit_transform(data)
    ##tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(df)
    #from bioinfokit.visuz import cluster
    #cluster.tsneplot(score=tsne_em)
    #color_class = df['class'].to_numpy()
    #cluster.tsneplot(score=tsne_em, colorlist=color_class, legendpos='upper right', legendanchor=(1.15, 1) )
    #
    #identifiers = []
    #labels = []
    #sequences = []
    #solubility = []
    #for record in SeqIO.parse(fasta_path, "fasta"):
    #    identifiers.append(record.id)
    #    sequences.append(str(record.seq))
    #    # df = df[df['solubility'] != 'U']
    #    df['length'] = df['seq'].apply(lambda x: len(x))
    #    print(df.describe())
    ###################################################################################################################
            

    #print("The first protein in the set has id {} and the embedding is of size {}."
    #    .format(proteins[0][0], len(proteins[0][1])))
    #print('The size is of the set is',len(proteins))
    #mapping_file = read_csv('mapping_file.csv', index_col=0)
    #
    ## Make sure the index is treated as a string, especially when using simple remapping in pipeline.
    #mapping_file.index = mapping_file.index.astype(str)
    #proteins = [(mapping_file.loc[p[0]].original_id, p[1]) for p in proteins]