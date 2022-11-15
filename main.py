from baseline_agents import RandomAgent, HighestDegreeAgent, LowestDegreeAgent, GreedyAgent
from DQN_agent import GNN, QN, DQNAgent
from helpers_simulation import *
from helpers_miscellaneous import *
from helpers_rewards import *


def main():
    base_path = '/content/drive/My Drive/GraphRL/Networks/'

    network_type = 'Synthetic'
    generator_type = 'ER'  # 'BA'

    feature = betti_numbers  # nx.average_clustering, betti_numbers, compressibility

    train_environments, val_environments, test_environments = load_environments(base_path,
                                                                                network_type,
                                                                                generator_type,
                                                                                feature)

    num_episodes = 10

    all_feature_values = simulate(RandomAgent, train_environments, num_episodes)
    feature_values_mean_random = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    all_feature_values = simulate(HighestDegreeAgent, train_environments, num_episodes)
    feature_values_mean_max_degree = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    all_feature_values = simulate(LowestDegreeAgent, train_environments, num_episodes)
    feature_values_mean_min_degree = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    all_feature_values = simulate(GreedyAgent, train_environments, num_episodes)
    feature_values_mean_greedy = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    hyperparameters = get_hyperparameters()

    embedding_module = GNN(hyperparameters)
    q_net = QN(hyperparameters)

    embedding_module_untrained = deepcopy(embedding_module)
    q_net_untrained = deepcopy(q_net)

    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None,
                     learn_every=None,
                     optimizer=None,
                     epsilon=1, epsilon_decay_rate=1, epsilon_min=1)

    all_feature_values = simulate(agent, train_environments, num_episodes=2)
    feature_values_mean_DQN = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    plt.title('Train Networks (Untrained DQN)')
    plt.xlabel('Step')
    plt.ylabel('Feature')
    plt.plot(feature_values_mean_random, label='Random', color='blue')
    plt.plot(feature_values_mean_max_degree, label='Max Degree', color='orange')
    plt.plot(feature_values_mean_min_degree, label='Min Degree', color='red')
    plt.plot(feature_values_mean_greedy, label='Greedy', color='green')
    plt.plot(feature_values_mean_DQN, label='DQN', color='black')
    plt.legend()
    # save_path = os.path.join(base_path, network_type, generator_type, 'Model', feature.__name__, 'Figures')
    # plt.savefig(os.path.join(save_path, 'pre_training_performance.eps'), format='eps')

    num_steps = 50000  # number of steps in each environment

    epsilon = hyperparameters['epsilon_initial']
    epsilon_min = hyperparameters['epsilon_min']
    epsilon_decay_rate = (2 * (epsilon_min - epsilon)) / num_steps

    discount_factor = hyperparameters['discount_factor']
    learning_rate = hyperparameters['learning_rate']

    hyperparameters['buffer_size'] = len(train_environments) * num_steps
    replay_buffer = ReplayBuffer(hyperparameters['buffer_size'])

    train_start = hyperparameters['train_start']
    batch_size = hyperparameters['batch_size']
    learn_every = hyperparameters['learn_every']

    optimizer = torch.optim.Adam([{'params': embedding_module.parameters()},
                                  {'params': q_net.parameters()}],
                                 lr=learning_rate)
    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer, train_start, batch_size,
                     learn_every,
                     optimizer,
                     epsilon, epsilon_decay_rate, epsilon_min)

    base_save_path = os.path.join(base_path, network_type, generator_type, 'Model', feature.__name__, 'Checkpoints')
    train_results, val_results = learn_environments(agent, train_environments, val_environments,
                                                    num_steps,
                                                    discount_factor,
                                                    base_save_path)

    plt.xlabel('Time (Arbitrary Units)')
    plt.ylabel('Validation Score')
    plt.plot(val_results['validation_scores'], label='DQN')
    plt.legend()
    # save_path = os.path.join(base_path, network_type, generator_type, 'Model', feature.__name__, 'Figures')
    # plt.savefig(os.path.join(save_path, 'val_score_training.eps'), format = 'eps')

    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None, learn_every=None,
                     optimizer=None,
                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)

    all_feature_values = simulate(agent, train_environments, num_episodes)
    feature_values_mean_DQN = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    plt.title('Train Performance')
    plt.xlabel('Step')
    plt.ylabel('Feature Value')
    plt.plot(feature_values_mean_random, label='Random', color='blue')
    plt.plot(feature_values_mean_max_degree, label='Max Degree', color='orange')
    plt.plot(feature_values_mean_min_degree, label='Min Degree', color='red')
    plt.plot(feature_values_mean_greedy, label='Greedy', color='green')
    plt.plot(feature_values_mean_DQN, label='DQN', color='black')
    plt.legend()
    # save_path = os.path.join(base_path, network_type, generator_type, 'Model', feature.__name__, 'Figures')
    # plt.savefig(os.path.join(save_path, 'post_training_performance.eps'), format = 'eps')

    all_feature_values = simulate(RandomAgent, test_environments, num_episodes)
    feature_values_mean_random = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    all_feature_values = simulate(HighestDegreeAgent, test_environments, num_episodes)
    feature_values_mean_max_degree = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    all_feature_values = simulate(LowestDegreeAgent, test_environments, num_episodes)
    feature_values_mean_min_degree = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    all_feature_values = simulate(GreedyAgent, test_environments, num_episodes)
    feature_values_mean_greedy = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    agent = DQNAgent(embedding_module, q_net,
                     replay_buffer=None, train_start=None, batch_size=None, learn_every=None,
                     optimizer=None,
                     epsilon=0, epsilon_decay_rate=None, epsilon_min=None)
    all_feature_values = simulate(agent, test_environments, num_episodes)
    feature_values_mean_DQN = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    agent = DQNAgent(embedding_module_untrained, q_net_untrained,
                     replay_buffer=None, train_start=None, batch_size=None, learn_every=None,
                     optimizer=None,
                     epsilon=1, epsilon_decay_rate=None, epsilon_min=None)
    all_feature_values = simulate(agent, test_environments, num_episodes)
    feature_values_mean_DQN_untrained = np.mean(np.mean(np.array(all_feature_values), axis=0), axis=0)

    plt.title('Test Performance')
    plt.xlabel('Step')
    plt.ylabel('Feature Value')
    plt.plot(feature_values_mean_random, label='Random', color='blue')
    plt.plot(feature_values_mean_max_degree, label='Max Degree', color='orange')
    plt.plot(feature_values_mean_min_degree, label='Min Degree', color='red')
    plt.plot(feature_values_mean_greedy, label='Greedy', color='green')
    plt.plot(feature_values_mean_DQN, label='DQN (trained)', color='black')
    plt.plot(feature_values_mean_DQN_untrained, label='DQN (untrained)', color='black', linestyle='dashed')
    plt.legend()
    # save_path = os.path.join(base_path, network_type, generator_type, 'Model', feature.__name__, 'Figures')
    # plt.savefig(os.path.join(save_path, 'test_performance.eps'), format = 'eps')


if __name__ == "__main__":
    main()
