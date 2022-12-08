import numpy as np
import pickle
from itertools import combinations
import time
import pylcs
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from tqdm.auto import trange
from shutil import copyfile
from os import mkdir
from os.path import isdir, exists
import h5py
import sys

# The following can help to automate
#clust_id = int(sys.argv[1])

def calc_dist(seq1, seq2):
    # Pattern matching with gestalt pattern matching
    seq1 = seq1[seq1>-1]
    seq1_str = ''.join(str(int(x)).zfill(4) for x in seq1)
    seq2 = seq2[seq2>-1]
    seq2_str = ''.join(str(int(x)).zfill(4) for x in seq2)
                                                          
    len_seq1 = int(math.floor(len(seq1_str)/4))
    len_seq2 = int(math.floor(len(seq2_str)/4))

    lcsstr = pylcs.lcs_string_length(seq1_str,seq2_str)
    km = int(math.floor(lcsstr)/4)
    similarity = (2*km)/(len_seq1+len_seq2)

    # 1 - similarity is the distance for the distance matrix
    return 1-similarity


with open("./01_succ_list/output.pickle", "rb") as f:
    data = pickle.load(f)
    npathways = len(data)
    lpathways = 600
    print(npathways)
    pathways = np.zeros((npathways,lpathways,5))
    for idx, val in enumerate(data):
        flipped_val = np.array(val)[::-1]
        for idx2, val2 in enumerate(flipped_val):            
            pathways[idx,idx2] = val2
    for pathway in pathways:
        # This is where discretization happens
        for step in pathway:
            if step[0] == 0:
                step[1] = -1
            elif step[2] > 69:
                step[1] = -1

    weights = []
    path_strings = []
    for pathway in pathways:
        nonzero = pathway[pathway[:,1]>-1]
        weights.append(nonzero[-1][4])
        path_strings.append(pathway[:,1])

    weights = np.array(weights)

    # You only have to generate the distance matrix once
    # (unless you change your discretization) so comment
    # out the following to save time when extracting
    # the clusters
    distmat =  pairwise_distances(X=path_strings, metric=lambda X, Y: calc_dist(X, Y))
    np.save("distmat.npy", distmat)

    # If the distmat is already made, just load it here to save time
    distmat = np.load("distmat.npy")

    distmat_condensed = squareform(distmat, checks=False)

    Z = sch.linkage(distmat_condensed, method='ward')

    # Set the horizontal threshold cutoff for the dendrogram
    dendrogram = sch.dendrogram(Z, no_labels=True, color_threshold=2)

    # Visualize the horizontal threshold cutoff for the dendrogram
    plt.axhline(y=2, c='k')
    plt.ylabel("distance")
    plt.xlabel("pathway")
    plt.savefig("dendrogram.pdf")

    # Set t to be the number of clusters you found in your dendrogram
    cluster_labels = sch.fcluster(Z, t=3, criterion="maxclust")

    # Use the following to make a separate h5 for all successfu pathways
    new_file = "west_succ.h5"
    # Use the following to make a separate h5 for each cluster
    #new_file = "west_succ_c0%s.h5"%clust_id

    if not exists(new_file):
        copyfile("../west.h5", new_file)

    first_iter=1
    with h5py.File("assign.h5", "r") as assign_file:
        last_iter = len(assign_file["nsegs"])

    tqdm_iter = trange(last_iter, first_iter - 1, -1, desc="iter")

    # Adjust depending on if you want to look at all of just one cluster
    trace_out_list = []
    #cluster = cluster_labels==clust_id
    data_arr = np.array(data)
    cluster_arr = data_arr
    #cluster_arr = data_arr[cluster]
    data_cl = list(cluster_arr)
    weights_cl = weights
    #weights_cl = weights[cluster]
    print(data_cl[np.argmax(weights_cl)][0])

    for idx, item in enumerate(data_cl):
        trace_out_list.append(list(np.array(item)[:,:2]))

    exclusive_set = {tuple(pair) for list in trace_out_list for pair in list}
    with h5py.File(new_file, "r+") as h5file:
        for n_iter in tqdm_iter:
            for n_seg in range(h5py.File("assign.h5", "r")["nsegs"][n_iter - 1]):
                if (n_iter, n_seg) not in exclusive_set:
                    h5file[f"iterations/iter_{n_iter:>08}/seg_index"]["weight", n_seg] = 0
