import numpy as np
import sys
import pickle

if len(sys.argv) !=2:
    print("USAGE> dataset_shuffle.py [meta_file]")

meta=pickle.load(open(sys.argv[1],'rb'))
meta_ids=list(meta.keys())
random_meta=np.random.choice(meta_ids,len(meta_ids),replace=False)

pickle.dump(random_meta,open(sys.argv[1][:-6]+'_shuffled_ids.list','wb'))
