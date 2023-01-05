import json
from GraphRL.environment import build_environments
from GraphRL.agent_DQN import GNN, QN, DQNAgent
from GraphRL.helpers_simulation import simulate
from GraphRL.helpers_miscellaneous import *


if __name__ == '__main__':

    run = 'synth_ER_med_betti'
    network_type = 'synthetic_BA'  # wikipedia, synthetic_ER, synthetic_BA
    size = 'small'  # size of dataset
    reward_function = nx.average_clustering  # betti_numbers, compressibility, nx.average_clustering

    base_path = '/Users/sppatankar/Developer/GraphRL/'
    run_path = base_path + 'Runs/' + network_type + '_' + size + '_' + reward_function.__name__ + '_run_' + run
    save_folder_path = os.path.join(run_path, 'Figures')
    create_save_folder(save_folder_path)

    baselines_path = os.path.join(base_path, 'Baselines/')
    baselines_filename = baselines_path + network_type + '_' + size + '_' + reward_function.__name__ + '_baselines.json'

    with open(baselines_filename, 'r') as f:
        baseline_agents_load = json.load(f)

    random_agent_train = baseline_agents_load['random']['train']
    max_degree_agent_train = baseline_agents_load['max_degree']['train']
    min_degree_agent_train = baseline_agents_load['min_degree']['train']
    greedy_agent_train = baseline_agents_load['greedy']['train']

    feature_values_mean_random_train = np.mean(np.mean(np.array(random_agent_train), axis=0), axis=0)
    feature_values_mean_max_degree_train = np.mean(np.mean(np.array(max_degree_agent_train), axis=0), axis=0)
    feature_values_mean_min_degree_train = np.mean(np.mean(np.array(min_degree_agent_train), axis=0), axis=0)
    feature_values_mean_greedy_train = np.mean(np.mean(np.array(greedy_agent_train), axis=0), axis=0)

    random_agent_val = baseline_agents_load['random']['val']
    max_degree_agent_val = baseline_agents_load['max_degree']['val']
    min_degree_agent_val = baseline_agents_load['min_degree']['val']
    greedy_agent_val = baseline_agents_load['greedy']['val']

    feature_values_mean_random_val = np.mean(np.mean(np.array(random_agent_val), axis=0), axis=0)
    feature_values_mean_max_degree_val = np.mean(np.mean(np.array(max_degree_agent_val), axis=0), axis=0)
    feature_values_mean_min_degree_val = np.mean(np.mean(np.array(min_degree_agent_val), axis=0), axis=0)
    feature_values_mean_greedy_val = np.mean(np.mean(np.array(greedy_agent_val), axis=0), axis=0)

    random_agent_test = baseline_agents_load['random']['test']
    max_degree_agent_test = baseline_agents_load['max_degree']['test']
    min_degree_agent_test = baseline_agents_load['min_degree']['test']
    greedy_agent_test = baseline_agents_load['greedy']['test']

    feature_values_mean_random_test = np.mean(np.mean(np.array(random_agent_test), axis=0), axis=0)
    feature_values_mean_max_degree_test = np.mean(np.mean(np.array(max_degree_agent_test), axis=0), axis=0)
    feature_values_mean_min_degree_test = np.mean(np.mean(np.array(min_degree_agent_test), axis=0), axis=0)
    feature_values_mean_greedy_test = np.mean(np.mean(np.array(greedy_agent_test), axis=0), axis=0)

    # DQN_path = run_path + network_type + '_' + size + '_' + reward_function.__name__ + '_run_' + run
    DQN_filename = os.path.join(run_path, 'log.json')

    with open(DQN_filename, 'r') as f:
        log = json.load(f)

    before_training = log['before_training']
    train_log = log['train_log']
    val_log = log['val_log']

    assert log['run'] == run, 'Mismatch in run ID.'
    assert log['network_type'] == network_type, 'Mismatch in network type.'
    assert log['size'] == size, 'Mismatch in network size.'
    assert log['reward_function'] == reward_function.__name__, 'Mismatch in reward function.'

    plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.plot(np.mean(np.array(train_log['returns']), axis=0), label='LDP')
    plt.legend()
    filename = 'avg_return_training.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')

    plt.figure()
    plt.xlabel('Step')
    plt.ylabel('Validation Score')
    plt.plot(val_log['validation_steps'], val_log['validation_scores'])
    plt.legend()
    filename = 'val_score_training.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')

    hyperparameters = get_hyperparameters()

    embedding_module = GNN(hyperparameters)
    q_net = QN(hyperparameters)

    max_index = np.argmax(val_log['validation_scores'])
    steps = val_log['validation_steps'][max_index]
    checkpoint_name = os.path.join(run_path, 'model_' + str(steps) + '.pt')

    train_log_cpt, val_log_cpt = load_checkpoint(checkpoint_name, embedding_module, q_net)

    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None,
                     learn_every=None,
                     optimizer=None,
                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)

    feature_values_mean_DQN_train_before = np.mean(np.mean(np.array(before_training), axis=0), axis=0)
    feature_values_mean_DQN_train_after = np.mean(np.mean(np.array(train_log_cpt['feature_values_train']), axis=0),
                                                  axis=0)

    plt.figure()
    plt.title('Train Networks')
    plt.xlabel('Step')
    plt.ylabel('Feature')
    plt.plot(feature_values_mean_random_train, label='Random', linestyle='dashed')
    plt.plot(feature_values_mean_max_degree_train, label='Max Degree', linestyle='dashed')
    plt.plot(feature_values_mean_min_degree_train, label='Min Degree', linestyle='dashed')
    plt.plot(feature_values_mean_greedy_train, label='Greedy', linestyle='dashed')
    plt.plot(feature_values_mean_DQN_train_before, label='DQN (before)', linestyle='dashed')
    plt.plot(feature_values_mean_DQN_train_after, label='DQN')
    plt.legend()
    filename = 'training_performance.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')

    feature_values_mean_DQN_val = np.mean(np.mean(np.array(val_log_cpt['feature_values_val'][-1]), axis=0), axis=0)

    plt.figure()
    plt.title('Validation Networks')
    plt.xlabel('Step')
    plt.ylabel('Feature')
    plt.plot(feature_values_mean_random_val, label='Random', linestyle='dashed')
    plt.plot(feature_values_mean_max_degree_val, label='Max Degree', linestyle='dashed')
    plt.plot(feature_values_mean_min_degree_val, label='Min Degree', linestyle='dashed')
    plt.plot(feature_values_mean_greedy_val, label='Greedy', linestyle='dashed')
    plt.plot(feature_values_mean_DQN_val, label='DQN')
    plt.legend()
    filename = 'validation_performance.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')

    env_path = os.path.join(base_path, 'Environments', network_type + '_' + size + '.json')

    with open(env_path, 'r') as f:
        all_data = json.load(f)

    test_data = all_data['test']

    feature_mode = 'LDP'  # random, LDP (local degree profile), or constant (= 1)
    steps_per_episode = 10  # steps in each episode; average KNOT session has ~9 unique node visits
    num_episodes = 100

    test_environments = build_environments(test_data, feature_mode, steps_per_episode, reward_function)
    all_feature_values = simulate(agent, test_environments, num_episodes)
    feature_values_mean_DQN_test = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    plt.figure()
    plt.title('Test Networks')
    plt.xlabel('Step')
    plt.ylabel('Feature')
    plt.plot(feature_values_mean_random_test, label='Random', linestyle='dashed')
    plt.plot(feature_values_mean_max_degree_test, label='Max Degree', linestyle='dashed')
    plt.plot(feature_values_mean_min_degree_test, label='Min Degree', linestyle='dashed')
    plt.plot(feature_values_mean_greedy_test, label='Greedy', linestyle='dashed')
    plt.plot(feature_values_mean_DQN_test, label='DQN')
    plt.legend()
    filename = 'testing_performance.eps'
    plt.savefig(os.path.join(save_folder_path, filename), format='eps')
