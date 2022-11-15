from functools import lru_cache
import torch
from torch_geometric import utils
import networkx as nx
import random
from copy import deepcopy
import matplotlib.pyplot as plt


@lru_cache(maxsize=100000)
def get_NX_subgraph(environment, frozen_set_of_nodes):
    return environment.graph_NX.subgraph(list(frozen_set_of_nodes))


@lru_cache(maxsize=500000)
def get_PyG_subgraph(environment, frozen_set_of_nodes):
    return environment.graph_PyG.subgraph(torch.tensor(list(frozen_set_of_nodes)))


@lru_cache(maxsize=100000)
def compute_feature_value(environment, state_subgraph_NX):
    return environment.feature_function(state_subgraph_NX)


@lru_cache(maxsize=100000)
def get_neighbors(environment, frozen_set_of_nodes, cutoff=1):
    """
    Returns the n-th degree neighborhood of a set of nodes, where degree
    is specified by the cutoff argument.
    """

    nodes = list(frozen_set_of_nodes)
    neighbors = set()

    for node in nodes:
        neighbors.update(set(nx.single_source_shortest_path_length(environment.graph_NX,
                                                                   node,
                                                                   cutoff=cutoff).keys()))

    neighbors = neighbors - set(nodes)  # remove input nodes from their own neighborhood

    if not neighbors:
        neighbors = set(environment.graph_NX.nodes()) - set(environment.visited)

    return list(neighbors)


class GraphEnvironment:

    def __init__(self, ID, graph_NX, feature):
        super().__init__()

        self.ID = ID  # identifier for the environment

        self.graph_NX = graph_NX  # environment graph (NetworkX Graph object)
        self.graph_PyG = utils.from_networkx(graph_NX, group_node_attrs=all)
        self.num_nodes = self.graph_NX.number_of_nodes()

        self.visited = [random.choice(list(self.graph_NX.nodes()))]  # list of visited nodes

        self.state_NX = get_NX_subgraph(self, frozenset(self.visited))
        self.state_PyG = get_PyG_subgraph(self, frozenset(self.visited))

        self.feature_function = feature  # function handle to network feature-of-interest

    def step(self, action):
        """
        Execute an action in the environment, i.e. visit a new node.
        """

        assert action in self.get_actions(self.visited), "Invalid action!"
        visited_new = deepcopy(self.visited)
        visited_new.append(action)  # add new node to list of visited nodes
        self.visited = visited_new
        self.state_NX = get_NX_subgraph(self, frozenset(self.visited))
        self.state_PyG = get_PyG_subgraph(self, frozenset(self.visited))
        reward = self.compute_reward()
        terminal = bool(len(self.visited) == self.graph_NX.number_of_nodes())

        return self.get_state_dict(), reward, terminal, self.get_info()

    def compute_reward(self):
        reward = compute_feature_value(self, self.state_NX)

        return reward

    def reset(self):
        """
        Reset to initial state.
        """

        self.visited = [random.choice(list(self.graph_NX.nodes()))]  # empty the list of visited nodes
        self.state_NX = get_NX_subgraph(self, frozenset(self.visited))
        self.state_PyG = get_PyG_subgraph(self, frozenset(self.visited))
        terminal = False

        return self.get_state_dict(), terminal, self.get_info()

    def get_state_dict(self):
        return {'visited': self.visited,
                'state_NX': self.state_NX,
                'state_PyG': self.state_PyG}

    def get_info(self):
        return {'environment_ID': self.ID,  # useful for DQN training
                'feature_value': compute_feature_value(self, self.state_NX)}

    def get_actions(self, nodes):
        """
        Returns available actions given a list of nodes.
        """

        return get_neighbors(self, frozenset(nodes))

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
