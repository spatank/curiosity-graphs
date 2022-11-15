import numpy as np
from numpy import inf, ix_
import networkx as nx
from ripser import ripser
import copy
import itertools
import random

import matplotlib.pyplot as plt


def make_filtration_matrix(G):
    """
    Takes in adjacency matrix and returns a filtration matrix for Ripser
    """

    N = G.shape[0]
    weighted_G = np.ones([N, N])
    for col in range(N):
        weighted_G[:col, col] = weighted_G[:col, col] * col
        weighted_G[col, :col] = weighted_G[col, :col] * col
    weighted_G += 1  # pushes second node's identifier to 2
    # removes diagonals, simultaneously resetting first node's identifier to 0
    weighted_G = np.multiply(G, weighted_G)
    # place 1 to N along the diagonal
    np.fill_diagonal(weighted_G, list(range(1, N + 1)))
    # set all zeros to be non-edges (i.e. at inf distance)
    weighted_G[weighted_G == 0] = np.inf
    # remove 1 from everywhere to ensure first node has identifier 0
    weighted_G -= 1

    return weighted_G


def betti_numbers(G, maxdim=2, dim=1):
    """
    Given a NetworkX graph object, computes number of topological cycles
    (i.e. Betti numbers) of various dimensions up to maxdim.
    """
    adj = nx.to_numpy_array(G)
    adj[adj == 0] = np.inf  # set unconnected nodes to be infinitely apart
    np.fill_diagonal(adj, 1)  # set diagonal to 1 to indicate all nodes are born at once
    bars = ripser(adj, distance_matrix=True, maxdim=maxdim)['dgms']  # returns barcodes
    bars_list = list(zip(range(maxdim + 1), bars))
    bettis_dict = dict([(dim, len(cycles)) for (dim, cycles) in bars_list])

    return bettis_dict[dim]  # return Betti number for dimension of interest


def get_barcode(filt_mat, maxdim=2):
    """
    Calculates the persistent homology for a given filtration matrix
    ``filt_mat``, default dimensions 0 through 2. Wraps ripser.
    """

    b = ripser(filt_mat, distance_matrix=True, maxdim=maxdim)['dgms']

    return list(zip(range(maxdim + 1), b))


def betti_curves(bars, length):
    """
    Takes in bars and returns the betti curves
    """

    bettis = np.zeros((len(bars), length))
    for i in range(bettis.shape[0]):
        bn = bars[i][1]
        for bar in bn:
            birth = int(bar[0])
            death = length + 1 if np.isinf(bar[1]) else int(bar[1] + 1)
            bettis[i][birth:death] += 1

    return bettis


def plot_bettis(bettis):
    N = bettis.shape[1]
    colors = ['xkcd:emerald green', 'xkcd:tealish', 'xkcd:peacock blue']
    for i in range(3):
        plt.plot(list(range(N)), bettis[i], color=colors[i],
                 label='$\\beta_{}$'.format(i),
                 linewidth=1)
    plt.xlabel('Nodes')
    plt.ylabel('Number of Cycles')
    plt.legend()


TOL = 1E-8


def repmat(M, m, n):
    return np.transpose(np.matlib.repmat(np.array([M]), m, n))


def rate_distortion_upper_info(G, setting=1):
    assert setting == 1, "Only setting 1 available."
    assert np.all(np.abs(G - G.T) < TOL), "Network must be symmetric."

    # network size
    N = len(G)
    E = np.sum(G) / 2

    # variables to save
    S = np.zeros(N)  # upper bound on entropy rate after clustering
    S_low = np.zeros(N)  # lower bound on entropy rate
    clusters = [[] for _ in range(N)]  # clusters[n] lists the nodes in each of the n clusters
    Gs = [[] for _ in range(N)]  # Gs[n] is the joint transition probability matrix for n clusters

    P_old = np.divide(G, np.transpose(np.matlib.repmat(np.array([np.sum(G, 1)]), N, 1)))

    # compute steady-state probabilities (works for undirected and possibly disconnected networks)
    p_ss = np.sum(G, 1) / np.sum(G)
    p_ss_old = copy.copy(p_ss)

    # compute initial entropy
    logP_old = np.log2(P_old, where=P_old > 0)
    logP_old[logP_old == -inf] = 0
    S_old = -np.sum(np.multiply(p_ss_old, np.sum(np.multiply(P_old, logP_old), 1)))
    # P_joint = np.multiply(P_old, np.transpose(np.matlib.repmat(p_ss_old, N, 1)))
    P_low = P_old

    # record initial values
    S[-1] = S_old
    S_low[-1] = S_old
    clusters[-1] = [[i] for i in range(N)]
    Gs[-1] = G

    for n in reversed(range(2, N)):
        pairs = np.array(list(itertools.combinations([k for k in range(1, n + 2)], 2)))
        I = pairs[:, 0]
        J = pairs[:, 1]

        # number of pairs
        num_pairs_temp = len(I)

        # track all entropies
        S_all = np.zeros(num_pairs_temp)

        for ind in range(num_pairs_temp):
            i = I[ind]
            j = J[ind]
            inds_not_ij = [v - 1 for v in list(range(1, i)) + list(range(i + 1, j)) + list(range(j + 1, n + 2))]

            # compute new stationary distribution
            p_ss_temp = [p_ss_old[inds_not_ij], p_ss_old[i - 1] + p_ss_old[j - 1]]

            # compute new transition probabilities
            P_temp_1 = np.sum(np.multiply(repmat(p_ss_old[inds_not_ij], 2, 1), P_old[ix_(inds_not_ij, [i - 1, j - 1])]),
                              1)
            P_temp_1 = np.divide(P_temp_1, p_ss_temp[:-1])

            P_temp_2 = np.sum(
                np.multiply(repmat(p_ss_old[[i - 1, j - 1]].T, n - 1, 1), P_old[ix_([i - 1, j - 1], inds_not_ij)]), 0)
            P_temp_2 = P_temp_2 / p_ss_temp[-1]

            P_temp_3 = np.sum(
                np.sum(np.multiply(repmat(p_ss_old[[i - 1, j - 1]], 2, 1), P_old[ix_([i - 1, j - 1], [i - 1, j - 1])])))
            P_temp_3 = P_temp_3 / p_ss_temp[-1]

            logP_temp_1 = np.log2(P_temp_1, where=P_temp_1 > 0)
            logP_temp_1[logP_temp_1 == -inf] = 0
            logP_temp_2 = np.log2(P_temp_2, where=P_temp_2 > 0)
            logP_temp_2[logP_temp_2 == -inf] = 0
            logP_temp_3 = np.array(np.log2(P_temp_3, where=P_temp_3 > 0))
            logP_temp_3[logP_temp_3 == -inf] = 0

            # compute change in upper bound on mutual information
            dS = - np.sum(np.multiply(np.multiply(p_ss_temp[:-1], P_temp_1), logP_temp_1))
            dS = dS - p_ss_temp[-1] * np.sum(np.multiply(P_temp_2, logP_temp_2))
            dS = dS - p_ss_temp[-1] * P_temp_3 * logP_temp_3
            dS = dS + np.sum(np.multiply(np.multiply(p_ss_old, P_old[:, i - 1]), logP_old[:, i - 1]))
            dS = dS + np.sum(np.multiply(np.multiply(p_ss_old, P_old[:, j - 1]), logP_old[:, j - 1]))
            dS = dS + p_ss_old[i - 1] * np.sum(np.multiply(P_old[i - 1, :], logP_old[i - 1, :]))
            dS = dS + p_ss_old[j - 1] * np.sum(np.multiply(P_old[j - 1, :], logP_old[j - 1, :]))
            dS = dS - p_ss_old[i - 1] * (
                    P_old[i - 1, i - 1] * logP_old[i - 1, i - 1] + P_old[i - 1, j - 1] * logP_old[i - 1, j - 1])
            dS = dS - p_ss_old[j - 1] * (
                    P_old[j - 1, j - 1] * logP_old[j - 1, j - 1] + P_old[j - 1, i - 1] * logP_old[j - 1, i - 1])

            S_temp = S_old + dS

            # track all entropies
            S_all[ind] = S_temp

        # find minimum entropy
        min_inds = np.where(S_all == min(S_all))
        iidx = random.randint(0, len(min_inds[0]) - 1)
        min_ind = [min_inds[0][iidx]]

        # save mutual information
        S_old = S_all[min_ind]
        S[n - 1] = S_old

        i_new = I[min_ind][0]
        j_new = J[min_ind][0]
        inds_not_ij = [v - 1 for v in
                       list(range(1, i_new)) + list(range(i_new + 1, j_new)) + list(range(j_new + 1, n + 2))]
        p_ss_new = np.concatenate((p_ss_old[inds_not_ij], np.array([p_ss_old[i_new - 1] + p_ss_old[j_new - 1]])))

        P_joint = np.multiply(repmat(p_ss_old.T, n + 1, 1), P_old)
        AB = P_joint[ix_(inds_not_ij, inds_not_ij)]
        A = np.expand_dims(np.sum(P_joint[ix_(inds_not_ij, [i_new - 1, j_new - 1])], 1), 1)
        B = np.expand_dims(np.sum(P_joint[ix_([i_new - 1, j_new - 1], inds_not_ij)], 0), 0)
        C = np.sum(np.sum(P_joint[ix_([i_new - 1, j_new - 1], [i_new - 1, j_new - 1])]))
        P_joint = np.block([[AB, A], [B, C]])

        P_old = np.divide(P_joint, repmat(p_ss_new.T, n, 1))
        p_ss_old = p_ss_new

        logP_old = np.log2(P_old, where=P_old > 0)
        logP_old[logP_old == -inf] = 0

        # record clusters and graph
        cls1_ind = [v - 1 for v in
                    list(range(1, i_new)) + list(range(i_new, (j_new - 1))) + list(range(j_new, (n + 1)))]
        cls1 = [clusters[n][v] for v in cls1_ind]
        cls2 = [list(clusters[n][i_new - 1]) + list(clusters[n][j_new - 1])]
        clusters[n - 1] = cls1 + cls2
        Gs[n - 1] = P_joint * 2 * E
        sp = np.expand_dims(P_low[:, i_new - 1] + P_low[:, j_new - 1], 1)
        P_low = np.concatenate((P_low[:, cls1_ind], sp), 1)
        logP_low = np.log2(P_low, where=P_low > 0)
        logP_low[logP_low == -inf] = 0
        S_low[n - 1] = -np.sum(np.multiply(p_ss, np.sum(np.multiply(P_low, logP_low), 1)))

    return S, S_low, clusters, Gs


def compressibility(graph_NX):
    """
    Computes network compressibility given a NetworkX graph object.
    """

    graph_NX_copy = nx.Graph(graph_NX)  # make an unfrozen copy
    for node in graph_NX_copy.nodes():  # add self-loops
        graph_NX_copy.add_edge(node, node)
    G = nx.to_numpy_array(graph_NX_copy)
    S, S_low, clusters, Gs = rate_distortion_upper_info(G)
    C = np.mean(S[-1] - S)

    return C
