import json
import pickle

from GraphRL.environment import GraphEnvironment, MultipleEnvironments
from GraphRL.agent_GNN import GNN, QN, DQNAgent
from GraphRL.helpers_rewards import *
from GraphRL.helpers_simulation import learn_environments
from GraphRL.helpers_miscellaneous import *

if __name__ == '__main__':
    run = '600000'  # for filenames during saving of results, always manually increment by 1
    network_type = 'Wikispeedia'  # wikipedia, synthetic_ER, synthetic_BA, Wikispeedia
    size = 'full'  # size of dataset
    feature_mode = 'LDP'  # random, LDP (local degree profile), or constant (= 1)
    reward_function = betti_numbers  # nx.average_clustering, betti_numbers, compressibility

    num_train_steps = 50000  # number of steps in each environment; ideally 50000
    val_every = 1000  # validate performance every val_every steps and save model

    base_path = '/Users/sppatankar/Developer/GraphRL/'
    run_path = base_path + 'Runs/' + network_type + '_' + size + '_' + \
               feature_mode + '_' + reward_function.__name__ + '_run_' + run
    create_save_folder(run_path)

    steps_per_episode = 10  # steps in each episode; average KNOT session has ~9 unique node visits

    # movies_adj_mat = np.load(os.path.join(base_path, 'MovieLens/edge_avg_50_adj.npy'))
    # movies_adj_mat = np.load(os.path.join(base_path, 'MovieLens/edge_avg_20_adj.npy'))
    # network_path = os.path.join(base_path, 'MovieLens/new_adj')
    network_path = os.path.join(base_path, 'Wikispeedia/wikispeedia_adj.pkl')
    with open(network_path, 'rb') as f:
        movies_adj_mat = pickle.load(f)
    movies_network = nx.convert_matrix.from_numpy_matrix(movies_adj_mat)
    G = node_featurizer(movies_network, mode='LDP')
    environments = []
    environment = GraphEnvironment(0, G, steps_per_episode, reward_function)
    environments.append(environment)
    train_environments = MultipleEnvironments(environments)
    val_environments = deepcopy(train_environments)

    hyperparameters = get_hyperparameters()
    embedding_module = GNN(hyperparameters)
    q_net = QN(hyperparameters)

    # print('Before:', compute_Frobenius_norm(embedding_module), compute_Frobenius_norm(q_net))
    # # Load pre-trained model:
    # load_model_run = '17'
    # load_model_path = base_path + 'Runs/' + network_type + '_' + size + '_' + \
    #                   feature_mode + '_' + reward_function.__name__ + '_run_' + load_model_run
    # with open(os.path.join(load_model_path, 'log.json'), 'r') as f:
    #     log = json.load(f)
    # max_index = np.argmax(log['validation_scores'])
    # steps = log['validation_steps'][max_index]
    # assert log['metadata']['run'] == load_model_run, 'Mismatch in run ID.'
    # assert log['metadata']['network_type'] == network_type, 'Mismatch in network type.'
    # assert log['metadata']['size'] == size, 'Mismatch in network size.'
    # assert log['metadata']['reward_function'] == reward_function.__name__, 'Mismatch'
    # load_checkpoint(os.path.join(load_model_path, 'model_' + str(steps) + '.pt'), embedding_module, q_net)
    # print('After:', compute_Frobenius_norm(embedding_module), compute_Frobenius_norm(q_net))

    epsilon = hyperparameters['epsilon_initial']
    epsilon_min = hyperparameters['epsilon_min']
    epsilon_decay_rate = (2 * (epsilon_min - epsilon)) / num_train_steps
    discount_factor = hyperparameters['discount_factor']
    learning_rate = hyperparameters['learning_rate']
    train_start = hyperparameters['train_start']
    batch_size = hyperparameters['batch_size']
    learn_every = hyperparameters['learn_every']
    replay_buffer = ReplayBuffer(hyperparameters['buffer_size'])
    optimizer = torch.optim.Adam([{'params': embedding_module.parameters()},
                                  {'params': q_net.parameters()}],
                                 lr=learning_rate)

    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer, train_start, batch_size,
                     learn_every,
                     optimizer,
                     epsilon, epsilon_decay_rate, epsilon_min)

    log = learn_environments(agent,
                             train_environments,
                             val_environments,
                             num_train_steps,
                             discount_factor,
                             val_every,
                             run_path)

    metadata = {'run': run, 'network_type': network_type, 'size': size, 'reward_function': reward_function.__name__}
    log['metadata'] = metadata

    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.plot(log['training_returns'], label='LDP')
    plt.legend()
    filename = 'avg_return_training.eps'
    plt.savefig(os.path.join(run_path, filename), format='eps')

    plt.figure()
    plt.xlabel('Step')
    plt.ylabel('Validation Score')
    plt.plot(log['validation_steps'], log['validation_scores'], label='LDP')
    plt.legend()
    filename = 'val_score_training.eps'
    plt.savefig(os.path.join(run_path, filename), format='eps')

    save_filename = os.path.join(run_path, 'log.json')

    with open(save_filename, 'w') as f:
        json.dump(log, f)
