import pandas as pd
import numpy as np
import gzip
import networkx as nx

eps = 1e-3
ti_network_file = '/home/hirak/Projects/CMSC828O/data/chilean_TI.txt'
df = pd.read_csv(ti_network_file,sep = '\t')

adj_df = df.loc[:,df.columns.isin(df.columns[2:])]
adj_mat = adj_df.values
G = nx.from_numpy_matrix(adj_mat)
print(nx.info(G))

num_classes, num_nodes = 4, G.number_of_nodes()
print(num_classes, num_nodes)

from sklearn.cluster import SpectralClustering

sc = SpectralClustering(num_classes, affinity='precomputed', n_init=100)
sc.fit(adj_mat)

# calculate tau
theta = np.zeros((num_classes,num_classes))
alpha = np.zeros(num_classes)
tau = np.zeros((num_nodes, num_classes))

for i,j in enumerate(sc.labels_):
    tau[i,j] = 1
# calculate theta
for (i,j) in G.edges:
    q,r = sc.labels_[i], sc.labels_[j]
    theta[q,r] += 1
    
tau += eps
tau = tau / tau.sum(axis=1, keepdims=1)
alpha = tau.sum(0)/ num_nodes

# E step
from tqdm import tqdm_notebook as tqdm
import sys
num_steps = int(sys.argv[1])


import time
L_old = 0
L_vec = []
for step in range(num_steps):
    start = time.time()
    # print('step ',step)
    tau_new_log = np.zeros((num_nodes, num_classes))
    for i in range(num_nodes):
        for q in range(num_classes):
            tau_new_log[i,q] = 0
            for j in range(num_nodes): 
                if(i != j):
                    # create prob
                    theta_q_tmp = theta[q,] + eps
                    theta_q_tmp = theta_q_tmp  / theta_q_tmp.sum()
                    theta_q_col = theta[:,q] + eps
                    theta_q_col = theta_q_col / theta_q_col.sum()

                    b_mat = (
                        adj_mat[i,j] * np.log( theta_q_tmp ) +
                        (1 - adj_mat[i,j]) * (np.log(1 - theta_q_tmp )) +
                        adj_mat[j,i] * np.log( theta_q_col ) +
                        (1 - adj_mat[i,j]) * (np.log( 1 - theta_q_col ))
                    )
                    #print(b_mat.shape, tau[j,].shape)

                    tau_new_log[i,q] += np.dot(tau[j,],b_mat)
            tau_new_log[i,q] += np.log(alpha[q])

    tau_new = np.exp(tau_new_log)
    tau = tau_new.copy()
    tau = tau / tau.sum(axis=1, keepdims=1)

    # M step
    # Calculate alpha
    alpha = tau.sum(axis = 0) / num_nodes
    # Calculate theta
    for q in range(num_classes):
        for r in range(num_classes):
            num = 0
            denom = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if ( i != j):
                        num += adj_mat[i,j] * tau[i,q] * tau[j,r]
                        denom +=  tau[i,q] * tau[j,r]
            theta[q,r] = num / denom
        theta[q,] = theta[q,] / theta[q,].sum()

    L = np.dot(tau.sum(axis = 0), np.log(alpha)) 
    for i in range(num_nodes):
        for j in range(num_nodes):
            if( i != j ):
                for q in range(num_classes):
                    # make theta prob distribution
                    theta_q_tmp = theta[q,] + eps
                    theta_q_tmp = theta_q_tmp / theta_q_tmp.sum()
                    for r in range(num_classes):
                        L += (0.5) * (
                            tau[i,q] * tau[j,r] *
                            (adj_mat[i,j] * np.log( theta_q_tmp[r] ) +
                            (1 - adj_mat[i,j]) * np.log(1 - theta_q_tmp[r] ))
                        )

    end = time.time()
    L_vec += [L]
    print('diff...{} elapsed time...{}'.format(L - L_old, end - start))
    L_old = L
