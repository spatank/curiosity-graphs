import json
# from GraphRL.environment import build_environments, build_wiki_environments
from GraphRL.environment import GraphEnvironment, MultipleEnvironments
from GraphRL.agent_DQN import GNN, QN, DQNAgent
from GraphRL.helpers_simulation import simulate
from GraphRL.helpers_miscellaneous import *
from GraphRL.helpers_rewards import *


if __name__ == '__main__':
    run = '60'
    network_type = 'Wikispeedia'  # wikipedia, synthetic_ER, synthetic_BA
    size = 'full'  # size of dataset
    feature_mode = 'LDP'  # random, LDP (local degree profile), or constant (= 1)
    reward_function = betti_numbers  # betti_numbers, compressibility, nx.average_clustering

    base_path = '/Users/sppatankar/Developer/GraphRL/'
    run_path = base_path + 'Runs/' + network_type + '_' + size + '_' + \
               feature_mode + '_' + reward_function.__name__ + '_run_' + run
    save_folder_path = os.path.join(run_path, 'Figures')
    create_save_folder(save_folder_path)

    # load baselines
    baselines_path = os.path.join(base_path, 'Baselines/')
    baselines_filename = baselines_path + network_type + '_' + size + '_' + reward_function.__name__ + '_baselines.json'

    with open(baselines_filename, 'r') as f:
        baseline_agents_load = json.load(f)

    random_agent = baseline_agents_load['random']
    max_degree_agent = baseline_agents_load['max_degree']
    min_degree_agent = baseline_agents_load['min_degree']
    greedy_agent = baseline_agents_load['greedy']

    feature_values_mean_random = np.mean(np.mean(np.array(random_agent), axis=0), axis=0)
    feature_values_mean_max_degree = np.mean(np.mean(np.array(max_degree_agent), axis=0), axis=0)
    feature_values_mean_min_degree = np.mean(np.mean(np.array(min_degree_agent), axis=0), axis=0)
    feature_values_mean_greedy = np.mean(np.mean(np.array(greedy_agent), axis=0), axis=0)

    with open(os.path.join(run_path, 'log.json'), 'r') as f:
        log = json.load(f)

    assert log['metadata']['run'] == run, 'Mismatch in run ID.'
    assert log['metadata']['network_type'] == network_type, 'Mismatch in network type.'
    assert log['metadata']['size'] == size, 'Mismatch in network size.'
    assert log['metadata']['reward_function'] == reward_function.__name__, 'Mismatch in reward function.'

    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.plot(log['training_returns'], label='LDP')
    plt.legend()
    filename = 'avg_return_training.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')

    plt.figure()
    plt.xlabel('Step')
    plt.ylabel('Validation Score')
    plt.plot(log['validation_steps'], log['validation_scores'])
    plt.legend()
    filename = 'val_score_training.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')

    steps_per_episode = 10  # steps in each episode; average KNOT session has ~9 unique node visits

    movies_adj_mat = np.load(os.path.join(base_path, 'MovieLens/edge_avg_20_adj.npy'))
    movies_network = nx.convert_matrix.from_numpy_matrix(movies_adj_mat)
    G = node_featurizer(movies_network, mode='LDP')
    environments = []
    environment = GraphEnvironment(0, G, steps_per_episode, reward_function)
    environments.append(environment)
    environments = MultipleEnvironments(environments)

    hyperparameters = get_hyperparameters()
    embedding_module = GNN(hyperparameters)
    q_net = QN(hyperparameters)

    # to simulate agent in the environments before training
    load_checkpoint(os.path.join(run_path, 'model_' + str(0) + '.pt'), embedding_module, q_net)

    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None,
                     learn_every=None,
                     optimizer=None,
                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)

    all_feature_values = simulate(agent, environments, num_episodes=1000)
    feature_values_mean_DQN_before_train = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    max_index = np.argmax(log['validation_scores'])
    steps = log['validation_steps'][max_index]

    # to simulate agent in the environments after training
    load_checkpoint(os.path.join(run_path, 'model_' + str(steps) + '.pt'), embedding_module, q_net)

    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None,
                     learn_every=None,
                     optimizer=None,
                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)

    all_feature_values = simulate(agent, environments, num_episodes=1000)
    feature_values_mean_DQN_after_train = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    plt.figure()
    plt.title('MovieLens Network')
    plt.xlabel('Step')
    plt.ylabel('Feature')
    plt.plot(feature_values_mean_random, label='Random', linestyle='dashed')
    plt.plot(feature_values_mean_max_degree, label='Max Degree', linestyle='dashed')
    plt.plot(feature_values_mean_min_degree, label='Min Degree', linestyle='dashed')
    plt.plot(feature_values_mean_greedy, label='Greedy', linestyle='dashed')
    plt.plot(feature_values_mean_DQN_after_train, label='DQN')
    plt.plot(feature_values_mean_DQN_before_train, label='DQN (before)', linestyle='dashed')
    plt.legend()
    filename = 'performance.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')
