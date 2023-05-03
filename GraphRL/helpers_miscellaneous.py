import os
import numpy as np
import torch  # get version using torch.__version__ for PyG wheels
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from tqdm import tqdm
import networkx as nx
from copy import deepcopy
from torch_geometric.nn import SAGEConv


def get_hyperparameters():
    parameters = {'num_node_features': 5,
                  'GNN_latent_dimensions': 64,
                  'embedding_dimensions': 64,
                  'QN_latent_dimensions': 32,
                  'buffer_size': 50000,
                  'train_start': 320,
                  'batch_size': 32,
                  'learn_every': 16,
                  'epsilon_initial': 1,
                  'epsilon_min': 0.1,
                  'discount_factor': 0.75,
                  'learning_rate': 3e-4}

    return parameters


def create_save_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print('Created folder:', folder_path)

    else:
        print('Folder exists. ')

    return


def generate_networks(num_networks, generator, verbose=False, **kwargs):
    networks = {}
    for i in tqdm(range(num_networks), disable=not verbose):
        graph = generator(**kwargs)
        while not nx.is_connected(graph):
            graph = generator(**kwargs)
        networks[i] = nx.node_link_data(graph)

    return networks


def initialize_weights(m):
    """
    Xavier's initialization of model weights.
    """

    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

    elif isinstance(m, SAGEConv):
        m.lin_l.weight.data = nn.init.xavier_uniform_(
            m.lin_l.weight.data, gain=nn.init.calculate_gain('relu'))

        if m.lin_l.bias is not None:
            m.lin_l.bias.data.zero_()

        m.lin_r.weight.data = nn.init.xavier_uniform_(
            m.lin_r.weight.data, gain=nn.init.calculate_gain('relu'))

        if m.lin_r.bias is not None:  # redundant
            m.lin_r.bias.data.zero_()

    elif isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))

        if m.bias is not None:
            m.bias.data.zero_()


def compute_Frobenius_norm(network):
    """
    Compute the Frobenius norm of all network tensors.
    """
    norm = 0.0

    for name, param in network.named_parameters():
        norm += torch.norm(param).data

    return norm.item()


def copy_parameters_from_to(source_network, target_network):
    """
    Update the parameters of the target network by copying values from the source
    network.
    """

    for source, target in zip(source_network.parameters(), target_network.parameters()):
        target.data.copy_(source.data)

    return


def average_area_under_the_curve(all_feature_values):
    """
    Returns the average area under the curve given a list of list of feature
    values. Each list inside all_feature_values corresponds to an environment.
    Each list inside that list corresponds to an episode. Each element of the
    inner list is a feature value at a given step during an episode.
    """

    all_areas = []
    for env_results in all_feature_values:
        areas = [sum(feature_values) for feature_values in env_results]
        all_areas.append(sum(areas) / len(areas))

    return sum(all_areas) / len(all_areas)


def generate_video(plotting_dict):
    feature_values_random = plotting_dict['random']
    feature_values_degree = plotting_dict['degree']
    feature_values_greedy = plotting_dict['greedy']
    feature_values_DQN = np.array(plotting_dict['DQN'])

    xlim = feature_values_DQN.shape[1]
    x = np.arange(xlim)  # number of nodes

    ylim = max(max(feature_values_random),
               max(feature_values_degree),
               max(feature_values_greedy),
               np.max(feature_values_DQN))

    fig, ax = plt.subplots()
    ax.axis([0, xlim, 0, ylim + 0.01 * ylim])

    _, = ax.plot(x, feature_values_random, label='random', color='blue')
    _, = ax.plot(x, feature_values_degree, label='max degree', color='orange')
    _, = ax.plot(x, feature_values_greedy, label='greedy', color='green')
    line4, = ax.plot([], [], label='DQN', color='black')

    ax.legend()

    plt.xlabel('Step')
    plt.ylabel('Value')

    def animate(i):
        line4.set_data(x, feature_values_DQN[i])

    anim_handle = animation.FuncAnimation(fig, animate,
                                          frames=len(feature_values_DQN),
                                          interval=100,
                                          blit=False, repeat=False,
                                          repeat_delay=10000)
    plt.close()  # do not show extra figure

    return anim_handle


def node_featurizer(graph_NX, mode='LDP'):
    graph_NX = deepcopy(graph_NX)

    attributes = {}

    for node in graph_NX.nodes():

        node_attributes = {}

        if mode == 'LDP':

            neighborhood = list(set([n for n in graph_NX.neighbors(node)]))

            if neighborhood:
                neighborhood_degrees = list(map(list, zip(*graph_NX.degree(neighborhood))))[1]
            else:  # no neighbors
                neighborhood_degrees = [0]

            node_attributes['feature_1'] = graph_NX.degree(node)
            node_attributes['feature_2'] = min(neighborhood_degrees)
            node_attributes['feature_3'] = max(neighborhood_degrees)
            node_attributes['feature_4'] = float(np.mean(neighborhood_degrees))
            node_attributes['feature_5'] = float(np.std(neighborhood_degrees))

        if mode == 'random':
            node_attributes['feature_1'] = random.random()
            node_attributes['feature_2'] = random.random()
            node_attributes['feature_3'] = random.random()
            node_attributes['feature_4'] = random.random()
            node_attributes['feature_5'] = random.random()

        if mode == 'constant':
            node_attributes['feature_1'] = 1.0
            node_attributes['feature_2'] = 1.0
            node_attributes['feature_3'] = 1.0
            node_attributes['feature_4'] = 1.0
            node_attributes['feature_5'] = 1.0

        attributes[node] = node_attributes

    nx.set_node_attributes(graph_NX, attributes)

    return graph_NX


def node_defeaturizer(graph_NX):
    graph_NX = deepcopy(graph_NX)

    attrs = set([k for n in graph_NX.nodes for k in graph_NX.nodes[n].keys()])

    for (n, d) in graph_NX.nodes(data=True):

        for attr in attrs:
            del d[attr]

    return graph_NX


class ReplayBuffer:

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.ptr = 0  # index to the latest experience in memory
        self.num_experiences = 0  # number of experiences stored in memory
        self.states = [None] * self.buffer_size
        self.actions = [None] * self.buffer_size
        self.next_states = [None] * self.buffer_size
        self.rewards = [None] * self.buffer_size
        self.discounts = [None] * self.buffer_size
        self.all_info = [None] * self.buffer_size

    def add(self, state_dicts, actions, next_state_dicts, rewards, discounts, all_info):

        # check if arguments are lists
        if not isinstance(state_dicts, list):  # i.e. a single experience
            state_dicts = [state_dicts]
            actions = [actions]
            next_state_dicts = [next_state_dicts]
            rewards = [rewards]
            discounts = [discounts]
            all_info = [all_info]

        for i in range(len(state_dicts)):
            self.states[self.ptr] = state_dicts[i]
            self.actions[self.ptr] = actions[i]
            self.next_states[self.ptr] = next_state_dicts[i]
            self.rewards[self.ptr] = rewards[i]
            self.discounts[self.ptr] = discounts[i]
            self.all_info[self.ptr] = all_info[i]

            if self.num_experiences < self.buffer_size:
                self.num_experiences += 1

            self.ptr = (self.ptr + 1) % self.buffer_size
            # if (ptr + 1) exceeds buffer size then begin overwriting older experiences

    def sample(self, batch_size):

        indices = np.random.choice(self.num_experiences, batch_size)
        states = [self.states[index] for index in indices]
        actions = [self.actions[index] for index in indices]
        next_states = [self.next_states[index] for index in indices]
        rewards = [self.rewards[index] for index in indices]
        discounts = [self.discounts[index] for index in indices]
        all_info = [self.all_info[index] for index in indices]

        return states, actions, next_states, rewards, discounts, all_info


def save_checkpoint(embedding_module, q_net,
                    step,
                    save_path):
    save_dict = {'embedding_module_state_dict': embedding_module.state_dict(),
                 'q_net_state_dict': q_net.state_dict(),
                 'step': step}
    torch.save(save_dict, save_path)

    return


def load_checkpoint(load_path, embedding_module, q_net):
    checkpoint = torch.load(load_path)
    embedding_module.load_state_dict(checkpoint['embedding_module_state_dict'])
    q_net.load_state_dict(checkpoint['q_net_state_dict'])

    return

