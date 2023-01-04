import random


class RandomAgent:
    """
    RandomAgent() chooses an action at random. The agent is not deterministic.
    """

    def __init__(self):
        super().__init__()

        self.environments = None  # should be an instance of MultipleEnvironments() class
        self.is_trainable = False  # useful to manage control flow during simulations

    def choose_action(self):

        if not self.environments:
            assert False, "Supply environment(s) for the agent to interact with."

        actions = []

        for environment in self.environments.environments:
            available_actions = environment.get_actions(environment.visited)
            action = random.choice(available_actions)
            actions.append(action)

        return actions


class HighestDegreeAgent:
    """
    HighestDegreeAgent() chooses the action with the highest node degree. The
    agent is deterministic.
    """

    def __init__(self):
        super().__init__()

        self.environments = None  # should be an instance of MultipleEnvironments() class
        self.is_trainable = False  # useful to manage control flow during simulations

    def choose_action(self):

        if not self.environments:
            assert False, "Supply environment(s) for the agent to interact with."

        actions = []

        for environment in self.environments.environments:
            available_actions = environment.get_actions(environment.visited)
            all_degrees = list(zip(*(environment.graph_NX.degree(available_actions))))[1]
            action_idx = all_degrees.index(max(all_degrees))  # first largest when ties
            action = available_actions[action_idx]
            actions.append(action)

        return actions


class LowestDegreeAgent:
    """
    LowestDegreeAgent() chooses the action with the lowest node degree. The
    agent is deterministic.
    """

    def __init__(self):
        super().__init__()

        self.environments = None  # should be an instance of MultipleEnvironments() class
        self.is_trainable = False  # useful to manage control flow during simulations

    def choose_action(self):

        if not self.environments:
            assert False, "Supply environment(s) for the agent to interact with."

        actions = []

        for environment in self.environments.environments:
            available_actions = environment.get_actions(environment.visited)
            all_degrees = list(zip(*(environment.graph_NX.degree(available_actions))))[1]
            action_idx = all_degrees.index(min(all_degrees))  # first smallest when ties
            action = available_actions[action_idx]
            actions.append(action)

        return actions


class GreedyAgent:
    """
    GreedyAgent() chooses the action that would result in the greatest reward.
    The agent uses a copy of the environment to simulate each available action and
    returns the best performing action. The agent is deterministic.
    """

    def __init__(self):
        super().__init__()

        self.environments = None  # should be an instance of MultipleEnvironments() class
        self.is_trainable = False  # useful to manage control flow during simulations

    def choose_action(self):

        if not self.environments:
            assert False, "Supply environment(s) for the agent to interact with."

        actions = []

        for environment in self.environments.environments:
            available_actions = environment.get_actions(environment.visited)
            best_reward = float('-inf')
            best_action = None

            for action in available_actions:
                state_dict, reward, terminal, info = environment.step(action)

                if reward > best_reward:
                    best_reward = reward
                    best_action = action

                environment.unstep()  # undo the action

            actions.append(best_action)

        return actions
