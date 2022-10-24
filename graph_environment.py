@lru_cache(maxsize = 10000)
def get_NX_subgraph(environment, frozen_set_of_nodes):

  return environment.graph_NX.subgraph(list(frozen_set_of_nodes))

@lru_cache(maxsize = 50000)
def get_PyG_subgraph(environment, frozen_set_of_nodes):

  return environment.graph_PyG.subgraph(torch.tensor(list(frozen_set_of_nodes)))

@lru_cache(maxsize = 10000)
def compute_feature_value(environment, state_subgraph_NX):

  return environment.feature_function(state_subgraph_NX)

@lru_cache(maxsize = 10000)
def get_neighbors(environment, frozen_set_of_nodes, cutoff = 1):
  """Returns the n-th degree neighborhood of a set of nodes, where degree 
  is specified by the cutoff argument.
  """

  nodes = list(frozen_set_of_nodes)
  neighbors = set()
  for node in nodes:
    neighbors.update(set(nx.single_source_shortest_path_length(environment.graph_NX, 
                                                               node, 
                                                               cutoff = cutoff).keys()))
  neighbors = neighbors - set(nodes) # remove input nodes from their own neighborhood

  return list(neighbors)

class GraphEnvironment():
  
  def __init__(self, graph_NX, feature, start_node = 0):
    super().__init__()

    self.graph_NX = graph_NX # environment graph (NetworkX Graph object)  
    self.graph_PyG = utils.from_networkx(graph_NX, group_node_attrs = all)
    self.num_nodes = self.graph_NX.number_of_nodes()

    self.start_node = start_node
    self.visited = [self.start_node] # list of visited nodes

    self.state_NX = get_NX_subgraph(self, frozenset(self.visited))
    self.state_PyG = get_PyG_subgraph(self, frozenset(self.visited))

    self.feature_function = feature # function handle to network feature-of-interest
    self.feature_values = [self.feature_function(self.state_NX)] # list to store values of the feature-of-interest
    
  def step(self, action):
    """Execute an action in the environment, i.e. visit a new node."""

    assert action in self.get_actions(self.visited), "Invalid action!"
    visited_new = deepcopy(self.visited)
    visited_new.append(action) # add new node to list of visited nodes
    self.visited = visited_new
    self.state_NX = get_NX_subgraph(self, frozenset(self.visited))
    self.state_PyG = get_PyG_subgraph(self, frozenset(self.visited))
    reward = self.compute_reward()
    terminal = bool(len(self.visited) == self.graph_NX.number_of_nodes())

    return self.get_state_dict(), reward, terminal, self.get_info()

  def compute_reward(self):

    self.feature_values.append(compute_feature_value(self, self.state_NX))
    reward = sum(self.feature_values)/len(self.visited)

    return reward

  def reset(self):
    """Reset to initial state."""

    self.visited = [self.start_node] # empty the list of visited nodes
    self.state_NX = get_NX_subgraph(self, frozenset(self.visited))
    self.state_PyG = get_PyG_subgraph(self, frozenset(self.visited))
    self.feature_values = [compute_feature_value(self, self.state_NX)]
    terminal = False

    return self.get_state_dict(), terminal, self.get_info()

  def get_state_dict(self):

    return {'visited': self.visited, 
            'state_NX': self.state_NX, 
            'state_PyG': self.state_PyG}
      
  def get_info(self):
    
    return {'feature_value': compute_feature_value(self, self.state_NX)}
  
  def get_actions(self, nodes):
    """ Returns available actions given a list of nodes.
    """

    return get_neighbors(self, frozenset(nodes))
  
  def render(self):
    """Render current state to the screen."""

    plt.figure()
    nx.draw(self.state_NX, with_labels = True)