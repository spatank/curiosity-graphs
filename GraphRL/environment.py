from functools import lru_cache
import random
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric import utils
from GraphRL.helpers_miscellaneous import node_featurizer, node_defeaturizer


# @lru_cache(maxsize=100000)
def get_NX_subgraph(frozen_set_of_nodes, graph_NX):
    return graph_NX.subgraph(list(frozen_set_of_nodes))


# @lru_cache(maxsize=500000)
def get_PyG_subgraph(frozen_set_of_nodes, graph_NX):
    subgraph_NX = get_NX_subgraph(frozen_set_of_nodes, graph_NX)
    subgraph_PyG = utils.from_networkx(subgraph_NX, group_node_attrs=all)

    return subgraph_PyG


# @lru_cache(maxsize=100000)
def compute_feature_value(feature_function, graph_NX):
    return feature_function(graph_NX)


# @lru_cache(maxsize=100000)
def get_neighbors(frozen_set_of_nodes, graph_NX):
    """
    Returns the neighborhood of a set of nodes.
    """

    nodes = list(frozen_set_of_nodes)
    all_neighbors = set()

    for node in nodes:
        neighbors = set([n for n in graph_NX.neighbors(node)])
        all_neighbors.update(neighbors)

    all_neighbors = all_neighbors - set(nodes)  # remove input nodes from their own neighborhood

    return list(all_neighbors)


class GraphEnvironment:

    def __init__(self, ID, graph_NX, steps_per_episode, feature):
        super().__init__()

        self.ID = ID  # identifier for the environment

        self.graph_NX = graph_NX  # environment graph (NetworkX Graph object)
        self.steps_per_episode = steps_per_episode
        self.num_nodes = self.graph_NX.number_of_nodes()
        self.visited = [random.choice(list(self.graph_NX.nodes()))]  # store a list of visited nodes
        self.state_NX = get_NX_subgraph(frozenset(self.visited), self.graph_NX)
        self.feature_function = feature  # handle to network feature-of-interest

    def step(self, action):
        """
        Execute an action in the environment, i.e. visit a new node.
        """

        assert action in self.get_actions(self.visited), "Invalid action!"
        self.visited = self.visited + [action]  # add new node to list of visited nodes
        self.state_NX = get_NX_subgraph(frozenset(self.visited), self.graph_NX)
        reward = self.compute_reward()
        terminal = self.is_terminal()

        return self.get_state_dict(), reward, terminal, self.get_info()

    def unstep(self):
        """
        Undo the most recent action. Multiple calls will undo multiple actions.
        """

        self.visited = self.visited[:-1]  # remove most recently visited node
        self.state_NX = get_NX_subgraph(frozenset(self.visited), self.graph_NX)

        return self.get_state_dict(), self.get_info()

    def is_terminal(self):
        case_1 = len(self.visited) == self.steps_per_episode
        case_2 = len(self.visited) == self.num_nodes

        return bool(case_1 or case_2)

    def compute_reward(self):
        return compute_feature_value(self.feature_function, self.state_NX)

    def reset(self):
        """
        Reset to initial state.
        """

        self.visited = [random.choice(list(self.graph_NX.nodes()))]  # empty the list of visited nodes
        self.state_NX = get_NX_subgraph(frozenset(self.visited), self.graph_NX)
        terminal = False

        return self.get_state_dict(), terminal, self.get_info()

    def get_state_dict(self):
        return {'visited': self.visited,
                'state_NX': self.state_NX}

    def get_info(self):
        return {'environment_ID': self.ID,  # useful for DQN training
                'feature_value': compute_feature_value(self.feature_function, self.state_NX)}

    def get_actions(self, visited_nodes):
        """
        Returns available actions given a list of visited nodes.
        """

        # get neighbors of most recently visited node
        actions = get_neighbors(frozenset(visited_nodes[-1:]), self.graph_NX)
        # remove the set of already visited nodes
        actions = set(actions) - set(visited_nodes)
        # if no neighbors are available to visit then allow jumps to distant nodes
        if not actions:
            actions = set(self.graph_NX.nodes()) - set(visited_nodes)

        return list(actions)

    def render(self):
        """
        Render current state to the screen.
        """

        plt.figure()
        nx.draw(self.state_NX, with_labels=True)


class MultipleEnvironments:

    def __init__(self, environments):

        self.environments = environments
        self.num_environments = len(self.environments)

    def reset(self):

        state_dicts = []
        terminals = []
        all_info = []

        for environment in self.environments:
            state_dict, terminal, info = environment.reset()
            state_dicts.append(state_dict)
            terminals.append(terminal)
            all_info.append(info)

        return state_dicts, terminals, all_info

    def step(self, actions):

        state_dicts = []
        rewards = []
        terminals = []
        all_info = []

        for idx, environment in enumerate(self.environments):
            state_dict, reward, terminal, info = environment.step(actions[idx])
            state_dicts.append(state_dict)
            rewards.append(reward)
            terminals.append(terminal)
            all_info.append(info)

        return state_dicts, rewards, terminals, all_info

    def __len__(self):
        return self.num_environments


def build_environments(network_data, feature_mode, steps_per_episode, reward_function):
    """
  Build graph environments from network_data. Node features can be local degree profile ('LDP'),
  'random', or 'constant'.
  """

    environments = []

    for idx in range(len(network_data)):
        base_G = nx.node_link_graph(network_data[str(idx)])
        base_G = node_defeaturizer(base_G)
        G = node_featurizer(base_G, mode=feature_mode)
        environment = GraphEnvironment(idx, G, steps_per_episode, reward_function)
        environments.append(environment)

    environments = MultipleEnvironments(environments)

    return environments
