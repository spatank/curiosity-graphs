import json
from GraphRL.environment import build_environments
from GraphRL.agent_DQN import GNN, QN, DQNAgent
from GraphRL.helpers_rewards import *
from GraphRL.helpers_simulation import learn_environments
from GraphRL.helpers_miscellaneous import *

if __name__ == '__main__':
    run = '50'  # for filenames during saving of results, always manually increment by 1
    network_type = 'synthetic_RG'  # wikipedia, synthetic_ER, synthetic_BA, synthetic_RG
    size = 'medium'  # size of dataset
    feature_mode = 'LDP'  # random, LDP (local degree profile), or constant (= 1)
    reward_function = compressibility  # nx.average_clustering, betti_numbers, compressibility

    num_train_steps = 50000  # number of steps in each environment; ideally 50000
    val_every = 1000  # validate performance every val_every steps and save model

    base_path = '/Users/sppatankar/Developer/GraphRL/'
    run_path = base_path + 'Runs/' + network_type + '_' + size + '_' + \
               feature_mode + '_' + reward_function.__name__ + '_run_' + run
    create_save_folder(run_path)

    data_load_path = os.path.join(base_path, 'Environments', network_type + '_' + size + '.json')
    with open(data_load_path, 'r') as f:
        all_data = json.load(f)

    train_data = all_data['train']
    val_data = all_data['val']
    test_data = all_data['test']

    steps_per_episode = 10  # steps in each episode; average KNOT session has ~9 unique node visits

    train_environments = build_environments(train_data, feature_mode, steps_per_episode, reward_function)
    val_environments = build_environments(val_data, feature_mode, steps_per_episode, reward_function)
    test_environments = build_environments(test_data, feature_mode, steps_per_episode, reward_function)

    hyperparameters = get_hyperparameters()
    embedding_module = GNN(hyperparameters)
    q_net = QN(hyperparameters)

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
