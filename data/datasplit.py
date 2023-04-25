from xml import dom
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train_cutoff = 0.95
    valid_cutoff = 0.97
    with open('/gaozhangyang/experiments/ProreinBinder/data/cath/cath-dataset-nonredundant-S40.list', 'r') as file:
        cathcodes = file.read()

    cathcodes = cathcodes.strip().split('\n')
    split = pd.DataFrame(columns=["domain", "chain", "CAT", "train", "valid", "test"])
    split["domain"] = cathcodes
    split["chain"] = [one[:5] for one in cathcodes]
    split["CAT"] = [one[:3] for one in cathcodes]

    L = len(cathcodes)
    train_cutoff = int(train_cutoff*L)
    valid_cutoff = int(valid_cutoff*L)

    clusters = {}
    for idx in range(split.shape[0]):
        CAT = split.loc[idx, "CAT"]
        domain = split.loc[idx, "domain"]
        if clusters.get(CAT) is None:
            clusters[CAT] = [domain]
        else:
            clusters[CAT].append(domain)

    train_inds = []
    valid_inds = []
    test_inds = []
    for key in sorted(clusters.keys()):
        cluster = clusters[key]
        if len(train_inds) + len(cluster) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(cluster) > valid_cutoff:
                test_inds += cluster
            else:
                valid_inds += cluster
        else:
            train_inds += cluster
    
    split = split.set_index("domain")
    split.loc[train_inds, "train"] = 1
    split.loc[valid_inds, "valid"] = 1
    split.loc[test_inds, "test"] = 1

    cmp_test_train = split.loc[test_inds, "CAT"].values.reshape(-1,1) == split.loc[train_inds, "CAT"].values
    cmp_valid_train = split.loc[valid_inds, "CAT"].values.reshape(-1,1) == split.loc[train_inds, "CAT"].values
    cmp_valid_test = split.loc[valid_inds, "CAT"].values.reshape(-1,1) == split.loc[test_inds, "CAT"].values

    assert not cmp_test_train.any()
    assert not cmp_valid_train.any()
    assert not cmp_valid_test.any()

    split.to_csv("./cath/CAT_split.csv")