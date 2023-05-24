import json
# from GraphRL.environment import build_environments, build_wiki_environments
from GraphRL.environment import GraphEnvironment, MultipleEnvironments
from GraphRL.agents_baseline import RandomAgent, HighestDegreeAgent, LowestDegreeAgent, GreedyAgent
from GraphRL.helpers_rewards import *
from GraphRL.helpers_simulation import simulate
from GraphRL.helpers_miscellaneous import *

if __name__ == '__main__':

    network_type = 'Wikispeedia'  # wikipedia, synthetic_ER, synthetic_BA
    size = 'full'  # size of dataset
    reward_function = betti_numbers  # betti_numbers, compressibility, nx.average_clustering

    base_path = '/Users/sppatankar/Developer/GraphRL/'
    save_path = os.path.join(base_path, 'Baselines/')
    save_filename = save_path + network_type + '_' + size + '_' + reward_function.__name__ + '_baselines.json'

    data_load_path = os.path.join(base_path, 'Environments', network_type + '_' + size + '.json')

    feature_mode = 'LDP'  # random, LDP (local degree profile), or constant (= 1)
    steps_per_episode = 10  # steps in each episode; average KNOT session has ~9 unique node visits
    num_episodes = 1000

    movies_adj_mat = np.load(os.path.join(base_path, 'MovieLens/edge_avg_20_adj.npy'))
    movies_network = nx.convert_matrix.from_numpy_matrix(movies_adj_mat)
    G = node_featurizer(movies_network, mode='LDP')
    environments = []
    environment = GraphEnvironment(0, G, steps_per_episode, reward_function)
    environments.append(environment)
    environments = MultipleEnvironments(environments)

    baseline_agents = {}

    print('\rRandom')
    agent = RandomAgent()
    baseline_agents['random'] = simulate(agent, environments, num_episodes)

    print('\rMax Degree')
    agent = HighestDegreeAgent()
    baseline_agents['max_degree'] = simulate(agent, environments, num_episodes)

    print('\rMin Degree')
    agent = LowestDegreeAgent()
    baseline_agents['min_degree'] = simulate(agent, environments, num_episodes)

    print('\rGreedy')
    agent = GreedyAgent()
    baseline_agents['greedy'] = simulate(agent, environments, num_episodes)

    with open(save_filename, 'w') as f:
        json.dump(baseline_agents, f)
