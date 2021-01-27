import numpy as np
import faiss
from time import time
import numpy as np


from collections import Counter
                    

class FaissKNeighbors:
    def __init__(self):
        self.index = None
        self.y = None

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X, k):
        distances, indices = self.index.search(X.astype(np.float32), k=k)
        votes = []
        for x in range(indices.shape[0]):
            votes_singel = []
            for y in range(indices.shape[1]):
                votes_singel += self.y[indices[x][y]]
            votes.append(np.asarray(votes_singel))
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
    def predict_prog(self, X,k):
        distances, indices = self.index.search(X.astype(np.float32), k=100)
        votes = []
        avg_count = 0
        print('start progessing')
        for x in range(indices.shape[0]):
            votes_singel = []
            counter = Counter({'0':0,'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0})
            tmp_count = 0
            for y in range(indices.shape[1]):
                
                avg_count+=1
                votes_singel += self.y[indices[x][y]]
                for s in self.y[indices[x][y]]:
                    counter[str(s)]+=1
                    tmp_count+=1
                if y >= k:                    
                    c = counter
                    #print(c)
                    #print(c.most_common(1))
                    most_commen = c.most_common(1)
                    if 1.0*most_commen[0][1]/tmp_count>=0.80:
                        #print('{}\t {} \t/ {}\t p: {}'.format(y,tmp_count,most_commen[0][1],1.0*most_commen[0][1]/tmp_count))
                        break
                    if x == 99:
                        print('{}\t {} \t/ {}\t p: {}'.format(y,tmp_count,most_commen[0][1],1.0*most_commen[0][1]/tmp_count))
                        votes_singel = votes_singel[:5]
                        avg_count-=95
                        break
            votes.append(np.asarray(votes_singel))
        print('AVG',avg_count,avg_count/indices.shape[0],indices.shape)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-h5", "--h5py_file", type=str,default='data/ec_vs_NOec_pide100_c50.h5', help="Path to data/ec_vs_NOec_pide100_c50.h5")
    parser.add_argument("-anno", "--annotations", type=str,default='data/annotations/merged_anno.txt', help="Path to data/annotations/merged_anno.txt")
    parser.add_argument("-br", "--breaking", type=int,default=-1 , help="Number of random samples after aquesitons stopps. (-1 == inf)")
    parser.add_argument("-k", "--k", type=int,default=5 , help="k-nn")
    parser.add_argument("-nonEC", "--nonEC", action="store_true", default=False, help="reduce embedings to (int), default value was 150 (0 is no reduction)")
    parser.add_argument("-nonEC_EC7", "--nonEC_EC7", action="store_true", default=False, help="reduce embedings to (int), default value was 150 (0 is no reduction)")
    
    
    #parser.add_argument("-br", "--breaking", type=int,default=-1 , help="Number of random samples after aquesitons stopps. (-1 == inf)")
    args = parser.parse_args()
    print(args)
   
    h5py_file = args.h5py_file
    anno = args.annotations

    print(h5py_file)
    #print('LOAD - This may take a while, if you run this the first time!')
    count = 0
    # First I want to see the splitt only in the 6 main-classes, befor we do something fancy here.
    
    from utils import join_h5_buffer,plot_multiclass_all,join_h5,loadBuffered,get_h5
    if args.nonEC_EC7:
        #Load h5 file
        identifiers,embeding = get_h5()
        #Load an other file (.txt / fasta)
        #anno_dic = loadBuffered(anno)        
        #Load h5 file and an other file that will be joined with, so you get aliged lists  
        #identifiers,embeding,annotations = join_h5_buffer('data/ec_vs_NOec_pide100_c50.h5','data/annotations/merged_anno.txt',store=True)
        #identifiers,embeding,other = join_h5(h5py_file,None,limit=args.breaking)
        anno_dic = loadBuffered(anno)
        def reduceAnno(key: str,anno_dic):
            if not key in anno_dic:
                anno_list = anno_dic[key].split('; ')
                annos = []
                for entry in anno_list:
                    annos.append(int(entry[0]))
                return annos
            i = int(anno_dic[key][0])
            return [i]
        color = [reduceAnno(i,anno_dic) for i in identifiers]
    elif args.nonEC:
        identifiers,embeding,_ = join_h5_buffer(h5py_file,None,limit=args.breaking,store=True)
        #identifiers,embeding,other = join_h5(h5py_file,None,limit=args.breaking)
        anno_dic = loadBuffered(anno)
        def reduceAnno(id: str):
            if id in anno_dic:
                return [1]
            else:
                return [0]
        color = [reduceAnno(i) for i in identifiers]
    else:
        identifiers,embeding,other = join_h5_buffer(h5py_file,anno,limit=args.breaking,store=True)
        
        #identifiers,embeding,other = join_h5(h5py_file,anno,limit=args.breaking)
        def reduceAnno(anno: str):
            anno_list = anno.split('; ')
            annos = []
            for entry in anno_list:
                annos.append(int(entry[0]))
            return annos
        color = [reduceAnno(c) for c in other]
    print('knn')
    knn = FaissKNeighbors()
    split = int(len(embeding)//5)
    knn.fit(np.asarray(embeding[:-split]),color[:-split])
    t0 = time()
    pred_val = knn.predict_prog(np.asarray(embeding[-split:]),args.k)
    def first_only(c):
        return c[0]
    val_label1 = np.asarray([first_only(c) for c in color[-split:]])
    t1 = time()
    print("%s: %.2g sec" % ('knn', t1 - t0))
    plot_multiclass_all(val_label1,pred_val)
    
    