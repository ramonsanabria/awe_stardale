import sys
import numpy as np
from tqdm import tqdm


feat_dir=sys.argv[1]
setname=sys.argv[2]
model_path=sys.argv[3]
nshard=int(sys.argv[4])

def get_counts(feat_dir, split, nshard):
    dict_counts = {}
    for r in range(nshard):
        label_path = f"{feat_dir}/{split}_{r}_{nshard}.labels"
        with open(label_path) as inputfile:
            for line in inputfile.readlines():
                for label in line.strip().split():
                    label_clean = label
                    if(label_clean not in dict_counts):
                        dict_counts[label_clean] = 1
                    else:
                        dict_counts[label_clean] = dict_counts[label_clean] + 1
    return dict_counts

def load_feature_shard(feat_dir, split, nshard, rank):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    label_path = f"{feat_dir}/{split}_{r}_{nshard}.labels"

    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            labels.append(line.strip().split())


    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip().split()[0]) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()
    
    feat = np.load(feat_path, mmap_mode="r")
    feat_list = [feat[offsets[i]: offsets[i] + lengs[i]] for i in range(len(lengs))]
    return feat_list, labels



dict_counts = get_counts(feat_dir, setname, nshard)
centroids = np.zeros((768,504))
    

for r in range(nshard):

    print("doing shard "+str(r)+" ...")
    feat_list, labels = load_feature_shard(feat_dir, setname, nshard, r)
    for idx, label_list in tqdm(enumerate(labels), total=len(labels)):
        for idx_inner, label in enumerate(label_list):
            centroids[:,int(label)]=centroids[:,int(label)] + (feat_list[idx][idx_inner,:]/dict_counts[label])


np.save(model_path, centroids)
