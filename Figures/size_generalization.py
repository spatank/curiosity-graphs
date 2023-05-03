from collections import defaultdict
import pickle
from helpers_figures import *
from GraphRL.helpers_simulation import simulate
from GraphRL.environment import build_environments
from GraphRL.agents_baseline import *
from GraphRL.agent_DQN import DQNAgent

if __name__ == '__main__':
    base_path = '/content/drive/My Drive/GraphRL_v2/Figure_3/'
    # settings
    feature_mode = 'LDP'
    reward_function = betti_numbers
    size = 'medium'
    # generation parameters
    N = 50  # number of nodes
    p = 0.2  # probability of edge creation (Erdos-Renyi)
    m = 4  # number of edges from new node (Barabasi-Albert)
    radius = 0.25  # (Random Geometric)
    k = 4  # (Watts-Strogatz)
    p_WS = 0.1  # (Watts-Strogatz)
    # experiment parameters
    num_networks = 10
    network_models = ['WS', 'ER', 'BA', 'RG']
    all_steps_per_episode = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    agents = ['random', 'max_degree', 'min_degree', 'greedy', 'DQN']

    all_data = {}

    for network_model in network_models:

        print(network_model)

        all_data[network_model] = defaultdict(list)

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

        for steps_per_episode in all_steps_per_episode:

            environments = build_environments(network_data, feature_mode, steps_per_episode, reward_function)

            for agent in agents:

                if agent == 'random':
                    agent = RandomAgent()
                    all_data[network_model]['random'].append(simulate(agent, environments))

                if agent == 'max_degree':
                    agent = HighestDegreeAgent()
                    all_data[network_model]['max_degree'].append(simulate(agent, environments))

                if agent == 'min_degree':
                    agent = LowestDegreeAgent()
                    all_data[network_model]['min_degree'].append(simulate(agent, environments))

                if agent == 'greedy':
                    agent = GreedyAgent()
                    all_data[network_model]['greedy'].append(simulate(agent, environments))

                if agent == 'DQN':
                    embedding_module, q_net = load_neural_network(base_path,
                                                                  network_model,
                                                                  size, feature_mode,
                                                                  reward_function)

                    # simulate DQN agent
                    agent = DQNAgent(embedding_module, q_net,
                                     replay_buffer=None, train_start=None, batch_size=None,
                                     learn_every=None,
                                     optimizer=None,
                                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)
                    all_data[network_model]['DQN'].append(simulate(agent, environments))

    with open(os.path.join(base_path, 'fig_3_homology.pkl'), 'wb') as f:
        pickle.dump(all_data, f)

    fontsize = 14

    plot_results_across_T(all_data, 'ER', all_steps_per_episode)
    plt.title('ER: Test Networks', fontsize=fontsize)
    plt.xlabel('Steps', fontsize=fontsize)
    plt.ylabel('Average Return', fontsize=fontsize)

    filename = 'ER_homology.eps'
    plt.savefig(os.path.join(base_path, filename), format='eps')

    plot_results_across_T(all_data, 'BA', all_steps_per_episode)
    plt.title('BA: Test Networks', fontsize=fontsize)
    plt.xlabel('Steps', fontsize=fontsize)
    plt.ylabel('Average Return', fontsize=fontsize)

    filename = 'BA_homology.eps'
    plt.savefig(os.path.join(base_path, filename), format='eps')

    plot_results_across_T(all_data, 'RG', all_steps_per_episode)
    plt.title('RG: Test Networks', fontsize=fontsize)
    plt.xlabel('Steps', fontsize=fontsize)
    plt.ylabel('Average Return', fontsize=fontsize)

    filename = 'RG_homology.eps'
    plt.savefig(os.path.join(base_path, filename), format='eps')

    plot_results_across_T(all_data, 'WS', all_steps_per_episode)
    plt.title('WS: Test Networks', fontsize=fontsize)
    plt.xlabel('Steps', fontsize=fontsize)
    plt.ylabel('Average Return', fontsize=fontsize)

    filename = 'WS_homology.eps'
    plt.savefig(os.path.join(base_path, filename), format='eps')
