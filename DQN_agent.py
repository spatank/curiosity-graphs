class GNN(nn.Module):

  def __init__(self, hyperparameters):
    super().__init__()
    
    self.conv1 = SAGEConv(
        hyperparameters['num_node_features'],
        hyperparameters['GNN_latent_dimensions'],
        aggr = 'mean')
    self.conv2 = SAGEConv(
        hyperparameters['GNN_latent_dimensions'],
        hyperparameters['embedding_dimensions'],
        aggr = 'mean')

  def forward(self, x, edge_index, batch = None):

    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = self.conv2(x, edge_index)
    x = F.relu(x) # node embeddings
    x = torch_geometric.nn.global_add_pool(x, batch = batch) # graph embedding

    return x

class QN(nn.Module):

  def __init__(self, hyperparameters):
    super().__init__()

    self.fc1 = nn.Linear(hyperparameters['embedding_dimensions'], 
                         hyperparameters['QN_latent_dimensions'])
    self.fc2 = nn.Linear(hyperparameters['QN_latent_dimensions'], 1)

  def forward(self, x):

    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)

    return x

class DQNAgent():

  def __init__(self, environment, 
               embedding_module, q_net, 
               replay_buffer, train_start, batch_size, 
               learn_every,
               optimizer, 
               epsilon, epsilon_decay_rate, epsilon_min):
    super().__init__()

    self.is_trainable = True # useful to manage control flow during simulations

    self.environment = environment

    self.embedding_module = embedding_module
    self.q_net = q_net

    self.target_embedding_module = deepcopy(embedding_module)
    self.target_q_net = deepcopy(q_net)
    
    # disable gradients for target networks
    for parameter in self.target_embedding_module.parameters():
      parameter.requires_grad = False
    for parameter in self.target_q_net.parameters():
      parameter.requires_grad = False
    
    self.replay_buffer = replay_buffer
    self.train_start = train_start # specify burn-in period
    self.batch_size = batch_size
    self.learn_every = learn_every # steps between updates to target nets

    self.optimizer = optimizer

    self.epsilon = epsilon # probability with which to select a non-greedy action
    self.epsilon_decay_rate = epsilon_decay_rate
    self.epsilon_min = epsilon_min

    self.step = 0 

  def choose_action(self):

    available_actions = self.environment.get_actions(self.environment.visited)

    new_subgraphs = [] # list to store all possible next states
    for action in available_actions:
      visited_nodes_new = deepcopy(self.environment.visited)
      visited_nodes_new.append(action)
      new_subgraph = get_PyG_subgraph(self.environment, frozenset(visited_nodes_new))
      new_subgraphs.append(new_subgraph)

    # create a batch to allow for a single forward pass
    batch = Batch.from_data_list(new_subgraphs)
    # gradients for the target networks are disabled
    with torch.no_grad(): 
      q_values = self.target_q_net(self.target_embedding_module(batch.x, 
                                                                batch.edge_index, 
                                                                batch.batch))

    if torch.rand(1) < self.epsilon: # explore
      action = np.random.choice(available_actions)
    else: # exploit
      action_idx = torch.argmax(q_values).item()
      action = available_actions[action_idx]

    return action

  def train(self, state_dict, action, next_state_dict, reward, discount):

    self.replay_buffer.add(state_dict, action, next_state_dict, reward, discount)
    self.step += 1

    if self.step < self.train_start: # inside the burn-in period
      return

    # (1) Get lists of experiences from memory
    states, actions, next_states, rewards, discounts = self.replay_buffer.sample(self.batch_size)
    
    # (2) Build state + action = new subgraph (technically identical to next state)
    new_subgraphs = []
    for idx, state_dict in enumerate(states):
      visited_nodes_new = deepcopy(state_dict['visited'])
      visited_nodes_new.append(actions[idx])
      assert visited_nodes_new == next_states[idx]['visited'], "train() assertion failed."
      new_subgraph = get_PyG_subgraph(self.environment, frozenset(visited_nodes_new))
      new_subgraphs.append(new_subgraph)
    batch = Batch.from_data_list(new_subgraphs)

    # (3) Pass batch of next_state subgraphs through ANN to get predicted q-values
    q_predictions = self.q_net(self.embedding_module(batch.x, 
                                                     batch.edge_index, 
                                                     batch.batch))

    # (4) Compute target q-values for batch
    q_targets = []
    for idx, next_state_dict in enumerate(next_states):
      available_actions = self.environment.get_actions(next_state_dict['visited'])
      if available_actions: # terminal states have no available actions
        new_subgraphs = [] # each available action results in a new state
        for action in available_actions:
          visited_nodes_new = deepcopy(next_state_dict['visited'])
          visited_nodes_new.append(action)
          new_subgraph = get_PyG_subgraph(self.environment, frozenset(visited_nodes_new))
          new_subgraphs.append(new_subgraph)
        batch = Batch.from_data_list(new_subgraphs)
        with torch.no_grad(): # technically, no_grad() is unnecessary
          q_target = self.target_q_net(self.target_embedding_module(batch.x, 
                                                                    batch.edge_index, 
                                                                    batch.batch))
          q_target = q_target.max().view(-1, 1) # get the largest next q-value
          q_target = rewards[idx] + discounts[idx] * q_target
          q_targets.append(q_target)
      else:
        q_targets.append(rewards[idx])
    q_targets = torch.Tensor(q_targets).view(-1, 1)
      
    # (5) Compute MSE loss between predicted and target q-values
    loss = F.mse_loss(q_predictions, q_targets).mean()

    # (6) Backpropagate gradients
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # (7) Copy parameters from source to target networks
    if self.step % self.learn_every == 0: 
      copy_parameters_from_to(self.embedding_module, self.target_embedding_module)
      copy_parameters_from_to(self.q_net, self.target_q_net)
      
    # (8) Decrease exploration rate
    self.epsilon *= self.epsilon_decay_rate
    self.epsilon = max(self.epsilon, self.epsilon_min)


class QNetworkAgent_Vanilla():

  def __init__(self, environment, 
               embedding_module, q_net, 
               optimizer, 
               epsilon, epsilon_decay_rate, epsilon_min):
    super().__init__()

    self.is_trainable = True # useful to manage control flow during simulations

    self.environment = environment
    self.embedding_module = embedding_module
    self.q_net = q_net
    self.optimizer = optimizer


    self.epsilon = epsilon # probability with which to select a non-greedy action
    self.epsilon_decay_rate = epsilon_decay_rate
    self.epsilon_min = epsilon_min

  def choose_action(self):

    available_actions = self.environment.get_actions(self.environment.visited)

    new_subgraph_list = [] # list to store all possible next states
    for action in available_actions:
      visited_nodes_new = deepcopy(self.environment.visited)
      visited_nodes_new.append(action)
      new_subgraph = self.environment.graph_PyG.subgraph(torch.tensor(visited_nodes_new))
      new_subgraph_list.append(new_subgraph)

    # create a batch to allow for a single forward pass
    new_subgraph_batch = Batch.from_data_list(new_subgraph_list)

    with torch.no_grad():
      q_values = self.q_net(self.embedding_module(new_subgraph_batch.x, 
                                                  new_subgraph_batch.edge_index, 
                                                  new_subgraph_batch.batch))
      
    if torch.rand(1) < self.epsilon: # explore
      action = np.random.choice(available_actions)
    else: # exploit
      action_idx = torch.argmax(q_values).item()
      action = available_actions[action_idx]

    return action

  def train(self, state_dict, action, next_state_dict, reward, discount):
    
    # (1) Build state + action (= next_state) subgraph 
    visited_nodes_new = deepcopy(state_dict['visited'])
    visited_nodes_new.append(action)
    assert visited_nodes_new == next_state_dict['visited'], "train() assertion failed."
    new_subgraph = self.environment.graph_PyG.subgraph(torch.tensor(visited_nodes_new))
    
    # (2) Pass next_state subgraph through ANN to get predicted q-value
    q_prediction = self.q_net(self.embedding_module(new_subgraph.x, 
                                                   new_subgraph.edge_index))
    
    # (3) Compute target q-value
    available_actions = self.environment.get_actions(next_state_dict['visited'])
    if available_actions: # states that are terminal have no available actions
      new_subgraphs = []
      for action in available_actions:
        visited_nodes_new = deepcopy(next_state_dict['visited'])
        visited_nodes_new.append(action)
        new_subgraph = self.environment.graph_PyG.subgraph(torch.tensor(visited_nodes_new))
        new_subgraphs.append(new_subgraph)

      batch = Batch.from_data_list(new_subgraphs)
      with torch.no_grad():
        q_target = self.q_net(self.embedding_module(batch.x, 
                                                    batch.edge_index, 
                                                    batch.batch))
        q_target = q_target.max().view(-1, 1)
        q_target = reward + discount * q_target
      
      # (4) Compute MSE loss between the predicted and target q-values
      loss = F.mse_loss(q_prediction, q_target)

      # (5) Backpropagate gradients
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # (6) Decrease exploration rate
      self.epsilon *= self.epsilon_decay_rate
      self.epsilon = max(self.epsilon, self.epsilon_min)