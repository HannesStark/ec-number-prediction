import numpy as np
import faiss
from time import time
import numpy as np
import sys

from collections import Counter
                    

class FaissKNeighbors:
    def __init__(self):
        self.index = None
        self.y = None
        self.max = 0
        self.max_reached = 0
        self.percent = 0.80

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
        distances, indices = self.index.search(X.astype(np.float32), k=1000)
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
                    self.max = max(y,self.max)
                if y >= k:                    
                    c = counter
                    #print(c)
                    #print(c.most_common(1))
                    most_commen = c.most_common(1)
                    if 1.0*most_commen[0][1]/tmp_count>=self.percent:
                        #print('{}\t {} \t/ {}\t p: {}'.format(y,tmp_count,most_commen[0][1],1.0*most_commen[0][1]/tmp_count))
                        break
                    if y == 99:
                        #print('{}\t {} \t/ {}\t p: {}'.format(y,tmp_count,most_commen[0][1],1.0*most_commen[0][1]/tmp_count))
                        votes_singel = votes_singel[:5]
                        avg_count-=95
                        self.max_reached+=1
                        break
            votes.append(np.asarray(votes_singel))
        print('AVG',avg_count,avg_count/indices.shape[0],indices.shape)
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

def delNonEC(identifiers,embeding,other, anno_dic, invert=False):
    dellist = []
    for i in range(len(identifiers)):
        if (not identifiers[i] in anno_dic and not invert) or (identifiers[i] in anno_dic and invert):
            sys.stdout.write('\r' + str(i))
            dellist.append(i)
            #print(i,len(identifiers))
    identifiers = np.delete(identifiers,dellist)
    embeding = np.delete(embeding,dellist,axis =0).reshape((-1,1024))
    other = np.delete(other,dellist)
    print()
    return identifiers,embeding,other


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-h5", "--h5py_file", type=str,default='data/ec_vs_NOec_pide100_c50.h5', help="Path to data/ec_vs_NOec_pide100_c50.h5")
    parser.add_argument("-anno", "--annotations", type=str,default='data/annotations/merged_anno.txt', help="Path to data/annotations/merged_anno.txt")
    parser.add_argument("-br", "--breaking", type=int,default=-1 , help="Number of random samples after aquesitons stopps. (-1 == inf)")
    parser.add_argument("-k", "--k", type=int,default=5 , help="k-nn")
    parser.add_argument("-nonEC", "--nonEC", action="store_true", default=False, help=" ")
    parser.add_argument("-nonEC_EC7", "--nonEC_EC7", action="store_true", default=False, help=" ")
    parser.add_argument("-train", "--train", action="store_true", default=False, help=" ")
    parser.add_argument("-first", "--first", action="store_true", default=False, help=" ")
    
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
    if args.train:
        identifiers,embeding,other = join_h5_buffer(h5py_file,'data/lookupSets/ec_vs_NOec_pide100_c50.fasta',store=True)
        identifiers_val,embeding_val,other_val = join_h5_buffer(h5py_file,'data/lookupSets/ec_vs_NOec_pide20_c50_val.fasta',store=True)
        identifiers_test,embeding_test,other_test = join_h5_buffer(h5py_file,'data/lookupSets/ec_vs_NOec_pide20_c50_test.fasta',store=True)
        anno_dic = loadBuffered(anno)
        print('befor',len(identifiers))
        identifiers,embeding,other = delNonEC(identifiers,embeding,other,identifiers_val,invert=True)
        identifiers,embeding,other = delNonEC(identifiers,embeding,other,identifiers_test,invert=True)
        print('after',len(identifiers))
        for i in identifiers:
            assert(not i in identifiers_val)
            assert(not i in identifiers_test)
        
        if args.nonEC:
            print(len(identifiers),len(identifiers_val))
            def reduceAnno(key: str,anno_dic):
                if key in anno_dic:
                   return [1]
                return [0]

            color = [reduceAnno(i,anno_dic) for i in identifiers]
            color_val = [reduceAnno(i,anno_dic) for i in identifiers_val]
            color_test = [reduceAnno(i,anno_dic) for i in identifiers_test]
        else:
            identifiers,embeding,other =delNonEC(identifiers,embeding,other, anno_dic)
            identifiers_val,embeding_val,other_val =delNonEC(identifiers_val,embeding_val,other_val, anno_dic)
            identifiers_test,embeding_test,other_test =delNonEC(identifiers_test,embeding_test,other_test, anno_dic)
            print('ec',len(identifiers),len(identifiers_val))
            def reduceAnno(key: str,anno_dic):
                if key in anno_dic:
                    anno_list = anno_dic[key].split('; ')
                    annos = []
                    for entry in anno_list:
                        annos.append(int(entry[0]))
                        if args.first:
                            break
                    return annos
                assert(False)
                return [0]

            color = [reduceAnno(i,anno_dic) for i in identifiers]
            color_val = [reduceAnno(i,anno_dic) for i in identifiers_val]
            color_test = [reduceAnno(i,anno_dic) for i in identifiers_test]

        ######
        print('knn')
        knn = FaissKNeighbors()
        print(embeding.shape)
        knn.fit(np.asarray(embeding),color)
        print([i for i in range(50,101,5)])
        def first_only(c):
            return c[0]
        best = 0
        best_ = ()
        for i in [i for i in range(50,101,5)]:
            t0 = time()
        
            knn.percent = i/100.
            #pred_val = knn.predict(np.asarray(embeding_val),1)
            pred_val = knn.predict_prog(np.asarray(embeding_val),args.k)
            #print('\t',i,'\t',knn.max_reached/split,'\t',knn.max)
            knn.max_reached = 0
            knn.max = 0
        
            val_label1 = np.asarray([first_only(c) for c in color_val])
            t1 = time()
            print("%s: %.2g sec" % ('knn', t1 - t0))
            _,_,a = plot_multiclass_all(val_label1,pred_val,show=False)
            if best< a:
                best = a
                best_ = i
        
        print('best model', best_)
        t0 = time()
        
        knn.percent = best_/100.
        #pred_test = knn.predict(np.asarray(embeding_test),1)
        pred_test = knn.predict_prog(np.asarray(embeding_test),args.k)
        #print('\t',i,'\t',knn.max_reached/split,'\t',knn.max)
        knn.max_reached = 0
        knn.max = 0
    
        val_label1 = np.asarray([first_only(c) for c in color_test])
        t1 = time()
        print("%s: %.2g sec" % ('knn', t1 - t0))
        _,_,a = plot_multiclass_all(val_label1,pred_test)
        
        
        exit()

    ###################################################################################################
    elif args.nonEC_EC7:
        #Load h5 file
        identifiers,embeding = get_h5()
        #Load an other file (.txt / fasta)
        #anno_dic = loadBuffered(anno)        
        #Load h5 file and an other file that will be joined with, so you get aliged lists  
        #identifiers,embeding,annotations = join_h5_buffer('data/ec_vs_NOec_pide100_c50.h5','data/annotations/merged_anno.txt',store=True)
        #identifiers,embeding,other = join_h5(h5py_file,None,limit=args.breaking)
        anno_dic = loadBuffered(anno)
        def reduceAnno(key: str,anno_dic):
            if key in anno_dic:
                anno_list = anno_dic[key].split('; ')
                annos = []
                for entry in anno_list:
                    annos.append(int(entry[0]))
                return annos
            i = 0
            return [i]
        color = [reduceAnno(i,anno_dic) for i in identifiers]

    ###################################################################################################
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
    ###################################################################################################
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

    ###################################################################################################
    print('knn')
    knn = FaissKNeighbors()
    split = int(len(embeding)//5)
    knn.fit(np.asarray(embeding[:-split]),color[:-split])
    print([i for i in range(50,101,5)])
    def first_only(c):
        return c[0]
    for i in [i for i in range(50,101,5)]:
        t0 = time()
    
        knn.percent = i/100.
        pred_val = knn.predict_prog(np.asarray(embeding[-split:]),args.k)
        #print('\t',i,'\t',knn.max_reached/split,'\t',knn.max)
        knn.max_reached = 0
        knn.max = 0
    
        val_label1 = np.asarray([first_only(c) for c in color[-split:]])
        t1 = time()
        print("%s: %.2g sec" % ('knn', t1 - t0))
        plot_multiclass_all(val_label1,pred_val)
    
    