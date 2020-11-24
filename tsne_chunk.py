from Bio.SeqIO import PirIO
import h5py
from time import time
import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.manifold import TSNE
import seaborn as sns
import math

def reduceAnno(anno: str):
        i = int(anno[0])
        return i
identifiers = []
embeding = []
color = []
def loadDataBase(args,reduceAnno = reduceAnno):
    print('LOAD - This may take a while, if you run this the first time!')
    h5py_file = args.h5py_file
    anno = args.annotations

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

def tsne(args,embeding,color,name=None):
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
    if name == None:
        plt.title('Samples {}; Iteration {}; Perplexity {}'.format(str(len(embeding)),str(args.n_iter),str(perplexity))) 
    else:
        plt.title('{}\nSamples {}; Iteration {}; Perplexity {}'.format(name,str(len(embeding)),str(args.n_iter),str(perplexity))) 
        
    if name != None:
        plt.savefig('img/'+name+'.png')
    if not args.no_show:
        plt.show()
    else:
        plt.clf()
    return Y

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-h5", "--h5py_file", type=str,default='data/ec_vs_NOec_pide100_c50.h5', help="Path to data/ec_vs_NOec_pide100_c50.h5")
    parser.add_argument("-anno", "--annotations", type=str,default='data/annotations/merged_anno.txt', help="Path to data/annotations/merged_anno.txt")
    parser.add_argument("-br", "--breaking", type=int,default=16000 , help="Number of random samples after aquesitons stopps. (-1 == inf)")
    parser.add_argument("-pp", "--perplexity", type=int,default=30 , help="perplexity")
    parser.add_argument("-ni", "--n_iter", type=int,default=1000 , help="n_iter")
    parser.add_argument("-no_show", "--no_show", action="store_true", default=False, help="Dont show plots")
    parser.add_argument("-in", "--imgname", type=str,default=None, help="")
    
    #parser.add_argument("-br", "--breaking", type=int,default=-1 , help="Number of random samples after aquesitons stopps. (-1 == inf)")
    args = parser.parse_args()
    print(args)

    loadDataBase(args)
    Y = tsne(args,embeding,color,name=args.imgname)
    identifiers_old = identifiers
    embeding_old = embeding
    color_old = color
    min0 = np.min(Y[:,0])
    w = (np.max(Y[:,0])-min0)/4.
    min1 = np.min(Y[:,1])
    h = (np.max(Y[:,1])-min1)/4.
    chunks = []
    for x in range(4):  
        chunks.append([])
        for y in range(4):
            chunk = ([],[],[]) # identifiers, embedding, color 
            chunks[-1].append(chunk)
    for i in range(Y.shape[0]):
        x = math.floor((Y[i,0]-min0)/w)
        x = min(3,x)
        y = math.floor((Y[i,1]-min1)/h)
        y = min(3,y)
        
        print(x,y)
        id, embeding, color = chunks[x][y]
        id.append(identifiers_old[i])
        embeding.append(embeding_old[i])
        color.append(color_old[i])
    for x in range(4):  
        for y in range(4):
            _, embeding, color = chunks[x][y]
            if len(embeding) <= 1:
                continue
            print(embeding)
            Y = tsne(args,embeding,color,name="{}; Chunk x = {}; y = {}".format(args.imgname,str(x),str(y)))
    


    #################################################################################################################
    # Source https://reneshbedre.github.io/blog/tsne.html
    # https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
    
