class RandomAgent():
  """RandomAgent() chooses an action at random. The agent is not deterministic."""
  
  def __init__(self, environment):
    super().__init__()
    
    self.environment = environment
    self.is_trainable = False # useful to manage control flow during simulations

  def choose_action(self):

    available_actions = self.environment.get_actions(self.environment.visited)
    action = random.choice(available_actions)

    return action

class HighestDegreeAgent():
  """HighestDegreeAgent() chooses the action with the highest node degree. The 
  agent is deterministic."""

  def __init__(self, environment):
    super().__init__()

    self.environment = environment
    self.is_trainable = False # useful to manage control flow during simulations

  def choose_action(self):

    available_actions = self.environment.get_actions(self.environment.visited)
    all_degrees = list(zip(*(self.environment.graph_NX.degree(available_actions))))[1]
    action_idx = all_degrees.index(max(all_degrees)) # first largest when ties
    action = available_actions[action_idx]

    return action

class GreedyAgent():
  """GreedyAgent() chooses the action that would result in the greatest reward.
  The agent uses a copy of the environment to simulate each available action and 
  returns the best performing action. The agent is deterministic."""

  def __init__(self, environment):
    super().__init__()

    self.environment = environment
    self.is_trainable = False # useful to manage control flow during simulations

  def choose_action(self):

    available_actions = self.environment.get_actions(self.environment.visited)

    best_reward = float('-inf')
    best_action = None

    for action in available_actions:

      environment_copy = deepcopy(self.environment)
      state_dict, reward, terminal, info = environment_copy.step(action)

      if reward > best_reward:
        best_reward = reward
        best_action = action

    return best_action