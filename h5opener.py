import h5py
from pandas import read_csv
import sys
if __name__ == "__main__":
    h5py_file= 'data/ec_vs_NOec_pide100_c50.h5'
    print(h5py_file)
    proteins = []
    print('LOAD - This may take a while, if you run this the first time!')
    count = 0
    with h5py.File(h5py_file, 'r') as f:
        first_key = list(f.keys())[0]
        print("The first protein in the set has id {} and the embedding is of size {}.".format(first_key, f[first_key].shape[0]))
        print('The size is of the set is',len(f.keys()))   
    #    l = len(f.keys())
    #    print(list(f['A0A140JWT2']))
    #    for protein_id in f.keys():
    #        #print('protein_id',protein_id)
    #        proteins.append((protein_id, list(f[protein_id])))
    #        count+=1
    #        sys.stdout.write('\r'+str(count)+"/"+str(l))
    #        #if count == 1000:
    #        #    print('Stop loading after 1000 protiens')
    #        #    break

            

    #print("The first protein in the set has id {} and the embedding is of size {}."
    #    .format(proteins[0][0], len(proteins[0][1])))
    #print('The size is of the set is',len(proteins))
    #mapping_file = read_csv('mapping_file.csv', index_col=0)
    #
    ## Make sure the index is treated as a string, especially when using simple remapping in pipeline.
    #mapping_file.index = mapping_file.index.astype(str)
    #proteins = [(mapping_file.loc[p[0]].original_id, p[1]) for p in proteins]