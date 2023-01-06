import json
from GraphRL.environment import build_environments
from GraphRL.agent_DQN import GNN, QN, DQNAgent
from GraphRL.helpers_rewards import *
from GraphRL.helpers_simulation import simulate, learn_environments
from GraphRL.helpers_miscellaneous import *


if __name__ == '__main__':

    run = 'synth_ER_med_betti_2'  # for filenames during saving of results
    network_type = 'synthetic_ER'  # wikipedia, synthetic_ER, synthetic_BA
    size = 'medium'  # size of dataset
    reward_function = betti_numbers  # nx.average_clustering, betti_numbers, compressibility

    num_train_steps = 50000  # number of steps in each environment; ideally 50000
    val_every = 1000  # validate performance every val_every steps and save model

    base_path = '/Users/sppatankar/Developer/GraphRL/'
    run_path = base_path + 'Runs/' + network_type + '_' + size + '_' + reward_function.__name__ + '_run_' + run
    create_save_folder(run_path)

    data_load_path = os.path.join(base_path, 'Environments', network_type + '_' + size + '.json')
    with open(data_load_path, 'r') as f:
        all_data = json.load(f)

    train_data = all_data['train']
    val_data = all_data['val']
    test_data = all_data['test']

    feature_mode = 'LDP'  # random, LDP (local degree profile), or constant (= 1)
    steps_per_episode = 10  # steps in each episode; average KNOT session has ~9 unique node visits

    train_environments = build_environments(train_data, feature_mode, steps_per_episode, reward_function)
    val_environments = build_environments(val_data, feature_mode, steps_per_episode, reward_function)
    test_environments = build_environments(test_data, feature_mode, steps_per_episode, reward_function)

    log = {'run': run, 'network_type': network_type, 'size': size, 'reward_function': reward_function.__name__}

    hyperparameters = get_hyperparameters()
    embedding_module = GNN(hyperparameters)
    q_net = QN(hyperparameters)

    # simulate agent in training environments before training
    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None,
                     learn_every=None,
                     optimizer=None,
                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)
    all_feature_values = simulate(agent, train_environments, num_episodes=100)
    feature_values_mean_DQN_before = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)
    log['before_training'] = all_feature_values

    epsilon = hyperparameters['epsilon_initial']
    epsilon_min = hyperparameters['epsilon_min']
    epsilon_decay_rate = (2 * (epsilon_min - epsilon)) / num_train_steps
    discount_factor = hyperparameters['discount_factor']
    learning_rate = hyperparameters['learning_rate']
    hyperparameters['buffer_size'] = len(train_environments) * num_train_steps
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

    train_log, val_log = learn_environments(agent,
                                            train_environments,
                                            val_environments,
                                            num_train_steps,
                                            discount_factor,
                                            val_every,
                                            run_path)

    log['train_log'] = train_log
    log['val_log'] = val_log

    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.plot(np.mean(np.array(train_log['returns']), axis=0), label='LDP')
    plt.legend()
    filename = 'avg_return_training.eps'
    plt.savefig(os.path.join(run_path, filename), format='eps')

    plt.figure()
    plt.xlabel('Step')
    plt.ylabel('Validation Score')
    plt.plot(val_log['validation_steps'], val_log['validation_scores'], label='LDP')
    plt.legend()
    filename = 'val_score_training.eps'
    plt.savefig(os.path.join(run_path, filename), format='eps')

    # simulate agent in training environments after training
    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None,
                     learn_every=None,
                     optimizer=None,
                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)
    all_feature_values = simulate(agent, train_environments, num_episodes=100)
    feature_values_mean_DQN_after = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)
    log['after_training'] = all_feature_values

    plt.figure()
    plt.title('Training Performance')
    plt.xlabel('Step')
    plt.ylabel('Feature')
    plt.plot(feature_values_mean_DQN_before, label='before', linestyle='dashed')
    plt.plot(feature_values_mean_DQN_after, label='after')
    plt.legend()
    filename = 'training_performance.eps'
    plt.savefig(os.path.join(run_path, filename), format='eps')

    save_filename = os.path.join(run_path, 'log.json')

    with open(save_filename, 'w') as f:
        json.dump(log, f)
