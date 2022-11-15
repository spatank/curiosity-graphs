import numpy as np
import networkx as nx
from ripser import ripser
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
