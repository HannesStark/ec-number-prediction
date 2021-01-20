import numpy as np
import faiss
from time import time
import numpy as np




class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        print(distances,indices)
        votes = self.y[indices]
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
    
    from utils import join_h5_buffer,plot_multiclass_all,join_h5,loadBuffered
    if args.nonEC_EC7:
        identifiers,embeding,other = join_h5_buffer(h5py_file,None,limit=args.breaking,store=True)
        #identifiers,embeding,other = join_h5(h5py_file,None,limit=args.breaking)
        anno_dic = loadBuffered(anno)
        def reduceAnno(key: str,anno_dic):
            if not key in anno_dic:
                return 0
            i = int(anno_dic[key][0])
            return i
        color = [reduceAnno(i,anno_dic) for i in identifiers]
    elif args.nonEC:
        identifiers,embeding,other = join_h5_buffer(h5py_file,None,limit=args.breaking,store=True)
        #identifiers,embeding,other = join_h5(h5py_file,None,limit=args.breaking)
        anno_dic = loadBuffered(anno)
        def reduceAnno(id: str):
            if id in anno_dic:
                return 1
            else:
                return 0
        color = [reduceAnno(i) for i in identifiers]
    else:
        identifiers,embeding,other = join_h5_buffer(h5py_file,anno,limit=args.breaking,store=True)
        #identifiers,embeding,other = join_h5(h5py_file,anno,limit=args.breaking)
        def reduceAnno(anno: str):
            i = int(anno[0])
            return i
        color = [reduceAnno(c) for c in other]
    print('knn')
    knn = FaissKNeighbors(k=args.k)
    split = int(len(embeding)//5)
    knn.fit(np.asarray(embeding[:-split]),np.asarray(color[:-split]))
    t0 = time()
    pred_val = knn.predict(np.asarray(embeding[-split:]))
    val_label1 = np.asarray(color[-split:])
    t1 = time()
    print("%s: %.2g sec" % ('knn', t1 - t0))
    plot_multiclass_all(val_label1,pred_val)
    
    