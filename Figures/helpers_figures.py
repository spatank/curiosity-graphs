import glob
import json
import time
from torch_geometric import utils
from GraphRL.agent_DQN import GNN, QN
from GraphRL.helpers_miscellaneous import *
from GraphRL.helpers_rewards import *


def load_neural_network(base_path, network_model, size, feature_mode, reward_function):
    hyperparameters = get_hyperparameters()
    embedding_module = GNN(hyperparameters)
    q_net = QN(hyperparameters)

    # load the training log
    files = glob.glob(base_path + 'Runs/' + 'synthetic_' + network_model + '_' + \
                      size + '_' + feature_mode + '_' + reward_function.__name__ + '*')

    run_path = files[0]

    with open(os.path.join(run_path, 'log.json'), 'r') as f:
        log = json.load(f)

    assert log['metadata']['network_type'] == 'synthetic_' + network_model, 'Mismatch in network type.'
    assert log['metadata']['size'] == size, 'Mismatch in network size.'
    assert log['metadata']['reward_function'] == reward_function.__name__, 'Mismatch in reward function.'

    # find and load the best performing model checkpoint
    max_index = np.argmax(log['validation_scores'])
    steps = log['validation_steps'][max_index]
    load_checkpoint(os.path.join(run_path, 'model_' + str(steps) + '.pt'), embedding_module, q_net)

    return embedding_module, q_net


def plot_results_across_T(all_data, network_type, all_steps_per_episode):
    avg_return_random, std_err_random = process_results_across_T(all_data, network_type,
                                                                 'random', all_steps_per_episode)
    avg_return_max_degree, std_err_max_degree = process_results_across_T(all_data, network_type,
                                                                         'max_degree', all_steps_per_episode)
    avg_return_min_degree, std_err_min_degree = process_results_across_T(all_data, network_type,
                                                                         'min_degree', all_steps_per_episode)
    avg_return_greedy, std_err_greedy = process_results_across_T(all_data, network_type,
                                                                 'greedy', all_steps_per_episode)
    avg_return_DQN, std_err_DQN = process_results_across_T(all_data, network_type,
                                                           'DQN', all_steps_per_episode)

    plt.figure()
    plt.plot(all_steps_per_episode, avg_return_random, label='Random', linestyle='dashed')
    plt.fill_between(all_steps_per_episode,
                     avg_return_random - std_err_random,
                     avg_return_random + std_err_random,
                     color='#1f77b4',
                     alpha=0.2)
    plt.plot(all_steps_per_episode, avg_return_max_degree, label='Max Degree', linestyle='dashed')
    plt.fill_between(all_steps_per_episode,
                     avg_return_max_degree - std_err_max_degree,
                     avg_return_max_degree + std_err_max_degree,
                     color='#ff7f0e',
                     alpha=0.2)
    plt.plot(all_steps_per_episode, avg_return_min_degree, label='Min Degree', linestyle='dashed')
    plt.fill_between(all_steps_per_episode,
                     avg_return_min_degree - std_err_min_degree,
                     avg_return_min_degree + std_err_min_degree,
                     color='#2ca02c',
                     alpha=0.2)
    plt.plot(all_steps_per_episode, avg_return_greedy, label='Greedy', linestyle='dashed')
    plt.fill_between(all_steps_per_episode,
                     avg_return_greedy - std_err_greedy,
                     avg_return_greedy + std_err_greedy,
                     color='#d62728',
                     alpha=0.2)
    plt.plot(all_steps_per_episode, avg_return_DQN, label='DQN')
    plt.fill_between(all_steps_per_episode,
                     avg_return_DQN - std_err_DQN,
                     avg_return_DQN + std_err_DQN,
                     color='#9467bd',
                     alpha=0.2)
    plt.legend()


def process_results_across_T(all_data, network_type, agent, all_steps_per_episode):
    all_average_returns = []
    standard_error = []

    for idx, steps_per_episode in enumerate(all_steps_per_episode):
        all_feature_values = np.array(all_data[network_type][agent][idx])
        all_returns = np.sum(all_feature_values, axis=2)
        average_return = np.mean(all_returns)
        all_average_returns.append(average_return)
        std_err = np.std(all_returns) / np.sqrt(all_returns.size)
        standard_error.append(std_err)

    return np.array(all_average_returns), np.array(standard_error)


def compute_wall_times(network_data,
                       reward_function, embedding_module, q_net,
                       feature_mode, num_iters=100):
    wall_times_greedy = []
    wall_times_GNN = []

    for idx, network in network_data.items():

        base_G = nx.node_link_graph(network)
        base_G = node_defeaturizer(base_G)
        G = node_featurizer(base_G, mode=feature_mode)
        G_PyG = utils.from_networkx(G, group_node_attrs=all)

        wall_times_iters_greedy = []
        wall_times_iters_GNN = []

        for _ in range(num_iters):
            # measure wall time for the greedy computation
            start_time = time.time()
            _ = reward_function(G)
            end_time = time.time()
            wall_time = end_time - start_time
            wall_times_iters_greedy.append(wall_time)

            # measure wall time for the GNN
            start_time = time.time()
            with torch.no_grad():
                _ = q_net(embedding_module(G_PyG.x, G_PyG.edge_index))
            end_time = time.time()
            wall_time = end_time - start_time
            wall_times_iters_GNN.append(wall_time)

        wall_times_greedy.append(np.mean(wall_times_iters_greedy))
        wall_times_GNN.append(np.mean(wall_times_iters_GNN))

    return wall_times_greedy, wall_times_GNN
