import json
from GraphRL.environment import build_environments
from GraphRL.agents_baseline import RandomAgent, HighestDegreeAgent, LowestDegreeAgent, GreedyAgent
from GraphRL.helpers_rewards import *
from GraphRL.helpers_simulation import simulate
from GraphRL.helpers_miscellaneous import *

if __name__ == '__main__':

    network_type = 'synthetic_WS'  # wikipedia, synthetic_ER, synthetic_BA, synthetic_RG, synthetic_WS
    size = 'large'  # size of dataset
    reward_function = compressibility  # betti_numbers, compressibility, nx.average_clustering

    base_path = '/Users/sppatankar/Developer/GraphRL/'
    save_path = os.path.join(base_path, 'Baselines/')
    save_filename = save_path + network_type + '_' + size + '_' + reward_function.__name__ + '_baselines.json'

    data_load_path = os.path.join(base_path, 'Environments', network_type + '_' + size + '.json')
    with open(data_load_path, 'r') as f:
        all_data = json.load(f)

    train_data = all_data['train']
    val_data = all_data['val']
    test_data = all_data['test']

    feature_mode = 'LDP'  # random, LDP (local degree profile), or constant (= 1)
    steps_per_episode = 10  # steps in each episode; average KNOT session has ~9 unique node visits
    num_episodes = 100

    train_environments = build_environments(train_data, feature_mode, steps_per_episode, reward_function)
    val_environments = build_environments(val_data, feature_mode, steps_per_episode, reward_function)
    test_environments = build_environments(test_data, feature_mode, steps_per_episode, reward_function)

    baseline_agents = {}
    random_agent = {}
    max_degree_agent = {}
    min_degree_agent = {}
    greedy_agent = {}

    print('\rRandom')
    agent = RandomAgent()
    all_feature_values_train = simulate(agent, train_environments, num_episodes)
    all_feature_values_val = simulate(agent, val_environments, num_episodes)
    all_feature_values_test = simulate(agent, test_environments, num_episodes)
    random_agent['train'] = all_feature_values_train
    random_agent['val'] = all_feature_values_val
    random_agent['test'] = all_feature_values_test
    baseline_agents['random'] = random_agent

    print('\rMax Degree')
    agent = HighestDegreeAgent()
    all_feature_values_train = simulate(agent, train_environments, num_episodes)
    all_feature_values_val = simulate(agent, val_environments, num_episodes)
    all_feature_values_test = simulate(agent, test_environments, num_episodes)
    max_degree_agent['train'] = all_feature_values_train
    max_degree_agent['val'] = all_feature_values_val
    max_degree_agent['test'] = all_feature_values_test
    baseline_agents['max_degree'] = max_degree_agent

    print('\rMin Degree')
    agent = LowestDegreeAgent()
    all_feature_values_train = simulate(agent, train_environments, num_episodes)
    all_feature_values_val = simulate(agent, val_environments, num_episodes)
    all_feature_values_test = simulate(agent, test_environments, num_episodes)
    min_degree_agent['train'] = all_feature_values_train
    min_degree_agent['val'] = all_feature_values_val
    min_degree_agent['test'] = all_feature_values_test
    baseline_agents['min_degree'] = min_degree_agent

    print('\rGreedy')
    agent = GreedyAgent()
    all_feature_values_train = simulate(agent, train_environments, num_episodes)
    all_feature_values_val = simulate(agent, val_environments, num_episodes)
    all_feature_values_test = simulate(agent, test_environments, num_episodes)
    greedy_agent['train'] = all_feature_values_train
    greedy_agent['val'] = all_feature_values_val
    greedy_agent['test'] = all_feature_values_test
    baseline_agents['greedy'] = greedy_agent

    with open(save_filename, 'w') as f:
        json.dump(baseline_agents, f)
