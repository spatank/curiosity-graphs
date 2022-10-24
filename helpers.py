def make_filtration_matrix(G):
    """
    Takes in adjacency matrix and returns a filtration matrix for Ripser
    """

    N = G.shape[0]
    weighted_G = np.ones([N, N])
    for col in range(N):
        weighted_G[:col, col] = weighted_G[:col, col] * col
        weighted_G[col, :col] = weighted_G[col, :col] * col
    weighted_G += 1 # pushes second node's identifier to 2
    # removes diagonals, simultaneously resetting first node's identifier to 0
    weighted_G = np.multiply(G, weighted_G) 
    # place 1 to N along the diagonal
    np.fill_diagonal(weighted_G, list(range(1, N + 1)))
    # set all zeros to be non-edges (i.e. at inf distance)
    weighted_G[weighted_G == 0] = np.inf
    # remove 1 from everywhere to ensure first node has identifier 0
    weighted_G -= 1
    
    return weighted_G

def betti_numbers(G, maxdim = 2, dim = 1):
  """
  Given a NetworkX graph object, computes number of topological cycles 
  (i.e. Betti numbers) of various dimensions upto maxdim.
  """
  adj = nx.to_numpy_array(G)
  adj[adj == 0] = np.inf # set unconnected nodes to be infinitely apart
  np.fill_diagonal(adj, 1) # set diagonal to 1 to indicate all nodes are born at once
  bars = ripser(adj, distance_matrix = True, maxdim = maxdim)['dgms'] # returns barcodes
  bars_list = list(zip(range(maxdim + 1), bars))
  bettis_dict = dict([(dim, len(cycles)) for (dim, cycles) in bars_list])

  return bettis_dict[dim] # return Betti number for dimension of interest

def get_barcode(filt_mat, maxdim = 2):
    """
    Calculates the persistent homology for a given filtration matrix
    ``filt_mat``, default dimensions 0 through 2. Wraps ripser.
    """

    b = ripser(filt_mat, distance_matrix = True, maxdim = maxdim)['dgms']

    return list(zip(range(maxdim + 1), b))

def betti_curves(bars, length):
    """
    Takes in bars and returns the betti curves
    """

    bettis = np.zeros((len(bars), length))
    for i in range(bettis.shape[0]):
        bn = bars[i][1]
        for bar in bn:
            birth = int(bar[0])
            death = length+1 if np.isinf(bar[1]) else int(bar[1]+1)
            bettis[i][birth:death] += 1

    return bettis

def plot_bettis(bettis):
  
  N = bettis.shape[1]
  colors = ['xkcd:emerald green', 'xkcd:tealish', 'xkcd:peacock blue']
  for i in range(3):
    plt.plot(list(range(N)), bettis[i], color = colors[i], 
             label = '$\\beta_{}$'.format(i), 
             linewidth = 1)
  plt.xlabel('Nodes')
  plt.ylabel('Number of Cycles')
  plt.legend()

def simulate(environment, agent, num_episodes = 100, verbose = True):
  """Simulate agent-environment interaction for a specified number of episodes."""
  
  states_NX = []
  states_PyG = []
  rewards = []
  feature_values = []
  steps = []

  for _ in tqdm(range(num_episodes), disable = not verbose):

    state_dict, terminal, info = environment.reset()

    episode_states_NX = [state_dict['state_NX']]
    episode_states_PyG = [state_dict['state_PyG']]
    episode_rewards = []
    episode_feature_values = []
    
    curr_step = 0

    while not terminal:

      action = agent.choose_action()
      state_dict, reward, terminal, info = environment.step(action)

      episode_states_NX.append(state_dict['state_NX'])
      episode_states_PyG.append(state_dict['state_PyG'])
      episode_rewards.append(reward)
      episode_feature_values.append(info['feature_value'])
      curr_step += 1
    
    states_NX.append(episode_states_NX)
    states_PyG.append(episode_states_PyG)
    rewards.append(episode_rewards)
    feature_values.append(episode_feature_values)
    steps.append(curr_step) # number of steps in the episode

  environment.reset()
  
  return states_NX, states_PyG, rewards, feature_values, steps

def learn_environment(environment, agent, num_steps, 
                      discount_factor = None, verbose = True):
  """Simulate agent-environment interaction for a number of steps specified 
  by num_steps and trains the agent if it is capable of being trained. Note that 
  the interaction ends after a finite number of steps regardless of whether the 
  ongoing episode has terminated."""

  all_episode_returns = []
  all_episode_feature_values = []

  episode_return = 0
  episode_feature_values = []
  state_dict, terminal, info = environment.reset()

  for step in tqdm(range(num_steps), disable = not verbose):

    action = agent.choose_action()
    next_state_dict, reward, terminal, info = environment.step(action)
    episode_return += reward
    episode_feature_values.append(info['feature_value'])

    if agent.is_trainable:
      discount = discount_factor * (1 - terminal)
      agent.train(state_dict, action, next_state_dict, reward, discount) 

    if terminal:
      all_episode_returns.append(episode_return)
      episode_return = 0
      all_episode_feature_values.append(episode_feature_values)
      episode_feature_values = []
      state_dict, terminal, info = environment.reset()
    else:
      state_dict = next_state_dict

  environment.reset()

  return all_episode_returns, all_episode_feature_values

def compute_Frobenius_norm(network):
    """
        Output: Frobenius norm of all network tensors
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

def generate_video(plotting_dict):

  feature_values_random = plotting_dict['random']
  feature_values_degree = plotting_dict['degree']
  feature_values_greedy = plotting_dict['greedy']
  feature_values_DQN = np.array(plotting_dict['DQN'])

  xlim = feature_values_DQN.shape[1]
  x = np.arange(xlim) # number of nodes

  ylim = max(max(feature_values_random), 
             max(feature_values_degree), 
             max(feature_values_greedy), 
             np.max(feature_values_DQN))

  fig, ax = plt.subplots()
  ax.axis([0, xlim, 0, ylim + 0.01 * ylim])

  line1, = ax.plot(x, feature_values_random, label = 'random', color = 'blue')
  line2, = ax.plot(x, feature_values_degree, label = 'max degree', color = 'orange')
  line3, = ax.plot(x, feature_values_greedy, label = 'greedy', color = 'green')
  line4, = ax.plot([], [], label = 'DQN', color = 'black')

  ax.legend()

  plt.xlabel('Step')
  plt.ylabel('Value')

  def animate(i):
    line4.set_data(x, feature_values_DQN[i])
    
  anim_handle = animation.FuncAnimation(fig, animate, 
                                        frames = len(feature_values_DQN),
                                        interval = 100,  
                                        blit = False, repeat = False, 
                                        repeat_delay = 10000)
  plt.close() # do not show extra figure

  return anim_handle

class ReplayBuffer():
  
  def __init__(self, buffer_size):

    self.buffer_size = buffer_size
    self.ptr = 0 # index to latest experience in memory
    self.num_experiences = 0 # number of experiences stored in memory
    self.states = [None] * self.buffer_size
    self.actions = [None] * self.buffer_size
    self.next_states = [None] * self.buffer_size
    self.rewards = [None] * self.buffer_size
    self.discounts = [None] * self.buffer_size

  def add(self, state, action, next_state, reward, discount):

    self.states[self.ptr] = state
    self.actions[self.ptr] = action
    self.next_states[self.ptr] = next_state
    self.rewards[self.ptr] = reward
    self.discounts[self.ptr] = discount
    
    if self.num_experiences < self.buffer_size:
      self.num_experiences += 1

    self.ptr = (self.ptr + 1) % self.buffer_size 
    # if (ptr + 1) exceeds buffer size then overwrite older experience

  def sample(self, batch_size):      

    indices = np.random.choice(self.num_experiences, batch_size)   

    states = [self.states[index] for index in indices] 
    actions = [self.actions[index] for index in indices] 
    next_states = [self.next_states[index] for index in indices] 
    rewards = [self.rewards[index] for index in indices] 
    discounts = [self.discounts[index] for index in indices] 
    
    return states, actions, next_states, rewards, discounts 

def node_featurizer(graph_NX):

  graph_NX = deepcopy(graph_NX)

  attributes = {}
  for node in graph_NX.nodes():
    
    neighborhood = set(nx.single_source_shortest_path_length(graph_NX, node, cutoff = 1).keys())
    neighborhood.remove(node) # remove node from its own neighborhood
    neighborhood = list(neighborhood) 
    neighborhood_degrees = list(map(list, zip(*graph_NX.degree(neighborhood))))[1]
    node_attributes = {}
    node_attributes['degree_1'] = graph_NX.degree(node)
    node_attributes['min_degree_1'] = min(neighborhood_degrees)
    node_attributes['max_degree_1'] = max(neighborhood_degrees)
    node_attributes['mean_degree_1'] = float(np.mean(neighborhood_degrees))
    node_attributes['std_degree_1'] = float(np.std(neighborhood_degrees))

    neighborhood = set(nx.single_source_shortest_path_length(graph_NX, node, cutoff = 2).keys())
    neighborhood.remove(node) # remove node from its own neighborhood
    neighborhood = list(neighborhood) 
    neighborhood_degrees = list(map(list, zip(*graph_NX.degree(neighborhood))))[1]
    node_attributes['min_degree_2'] = min(neighborhood_degrees)
    node_attributes['max_degree_2'] = max(neighborhood_degrees)
    node_attributes['mean_degree_2'] = float(np.mean(neighborhood_degrees))
    node_attributes['std_degree_2'] = float(np.std(neighborhood_degrees))

    attributes[node] = node_attributes
    
  nx.set_node_attributes(graph_NX, attributes)

  return graph_NX

def node_defeaturizer(graph_NX):

  graph_NX = deepcopy(graph_NX)

  for (n, d) in graph_NX.nodes(data = True):
    del d["degree_1"]
    del d["min_degree_1"]
    del d["max_degree_1"]
    del d["mean_degree_1"]
    del d["std_degree_1"]
    del d["min_degree_2"]
    del d["max_degree_2"]
    del d["mean_degree_2"]
    del d["std_degree_2"]

    return graph_NX