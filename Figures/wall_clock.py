from helpers_figures import *

if __name__ == '__main__':
    base_path = '/Users/lcaciagl/Developer/GraphRL/Figures/Figure_3/'
    network_model = 'RG'
    size = 'medium'
    feature_mode = 'LDP'
    num_networks = 50
    p = 0.2  # probability of edge creation (Erdos-Renyi)
    m = 4  # number of edges from new node (Barabasi-Albert)
    radius = 0.25  # (Random Geometric)
    k = 4  # (Watts-Strogatz)
    p_WS = 0.1  # (Watts-Strogatz)

    # embedding_module_homology, q_net_homology = load_neural_network(base_path,
    #                                                                 network_model,
    #                                                                 size,
    #                                                                 feature_mode,
    #                                                                 betti_numbers)

    embedding_module_compressibility, q_net_compressibility = load_neural_network(
        base_path, network_model, size, feature_mode, compressibility)

    all_Ns = list(range(5, 55, 5))

    all_networks = {}

    # times_greedy_homology = []
    # times_GNN_homology = []

    times_greedy_compressibility = []
    times_GNN_compressibility = []

    for N in tqdm(all_Ns):

        if network_model == 'ER':
            network_data = generate_networks(num_networks,
                                             nx.erdos_renyi_graph, n=N, p=p)
        if network_model == 'BA':
            network_data = generate_networks(num_networks,
                                             nx.barabasi_albert_graph, n=N, m=m)
        if network_model == 'RG':
            network_data = generate_networks(num_networks,
                                             nx.random_geometric_graph, n=N, radius=radius)
        if network_model == 'WS':
            network_data = generate_networks(num_networks,
                                             nx.watts_strogatz_graph, n=N, k=k, p=p_WS)

        # times_greedy_at_N, times_GNN_at_N = compute_wall_times(network_data,
        #                                                        betti_numbers,
        #                                                        embedding_module_homology,
        #                                                        q_net_homology,
        #                                                        feature_mode)
        # times_greedy_homology.append(times_greedy_at_N)
        # times_GNN_homology.append(times_GNN_at_N)

        times_greedy_at_N, times_GNN_at_N = compute_wall_times(network_data,
                                                               compressibility,
                                                               embedding_module_compressibility,
                                                               q_net_compressibility,
                                                               feature_mode)

        times_greedy_compressibility.append(times_greedy_at_N)
        times_GNN_compressibility.append(times_GNN_at_N)

    fontsize = 14

    # plt.figure()
    # plt.title('RG: Test Networks', fontsize=fontsize)
    # plt.xlabel('Nodes', fontsize=fontsize)
    # plt.ylabel('Wall Time', fontsize=fontsize)
    # plt.plot(list(range(5, 55, 5)), np.mean(np.array(times_GNN_homology), axis=1),
    #          '-o', color='#1f77b4', label='DQN-Homology')
    # plt.plot(list(range(5, 55, 5)), np.mean(np.array(times_greedy_homology), axis=1),
    #          '-x', color='#d62728', label='Greedy-Homology')
    # plt.legend()
    # filename = 'RG_homology_wall_time.eps'
    # plt.savefig(os.path.join(base_path, filename), format='eps')

    plt.figure()
    plt.title('RG: Test Networks', fontsize=fontsize)
    plt.xlabel('Nodes', fontsize=fontsize)
    plt.ylabel('Wall Time', fontsize=fontsize)
    plt.plot(list(range(5, 55, 5)), np.mean(np.array(times_GNN_compressibility), axis=1),
             '-o', color='#1f77b4', label='DQN-Compressibility')
    plt.plot(list(range(5, 55, 5)), np.mean(np.array(times_greedy_compressibility), axis=1),
             '-x', color='#d62728', label='Greedy-Compressibility')
    plt.legend()
    filename = 'RG_compressibility_wall_time.eps'
    plt.savefig(os.path.join(base_path, filename), format='eps')
