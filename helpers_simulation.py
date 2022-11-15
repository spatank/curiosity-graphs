from copy import deepcopy
from tqdm import tqdm
from helpers_miscellaneous import average_area_under_the_curve, save_checkpoint
import os


def simulate(agent, environments, num_episodes=100, verbose=True):
    """
    Simulate agent in multiple environments for a specified number of episodes.
    We do not use methods from the MultipleEnvironment() class because each
    environment may have a different number of nodes.
    """

    agent = deepcopy(agent)  # do not alter the original agent's environments
    agent.environments = environments  # supply the agent with different environments

    all_feature_values = []

    for idx, environment in enumerate(tqdm(environments.environments,
                                           disable=not verbose)):

        state_dict, terminal, info = environment.reset()
        environment_feature_values = []

        for _ in range(num_episodes):
            episode_feature_values = []

            while not terminal:
                actions = agent.choose_action()
                action = actions[idx]  # agent chooses an action for each environment
                state_dict, reward, terminal, info = environment.step(action)
                episode_feature_values.append(info['feature_value'])

            state_dict, terminal, info = environment.reset()  # reset environment after use

            environment_feature_values.append(episode_feature_values)

        all_feature_values.append(environment_feature_values)

    environments.reset()

    return all_feature_values


def learn_environments(agent, train_environments, val_environments,
                       num_steps, discount_factor, base_save_path,
                       log_val_results=True, verbose=True):
    """
    Train agent on multiple environments by simulating agent-environment
    interactions for a specified number of steps.
    """

    agent.environments = train_environments  # supply the agent with environments

    # training logs
    all_episode_returns_train = [[] for _ in range(train_environments.num_environments)]
    all_episode_feature_values_train = [[] for _ in range(train_environments.num_environments)]
    episode_returns_train = [0] * train_environments.num_environments
    episode_feature_values_train = [[] for _ in range(train_environments.num_environments)]

    # validation logs
    all_episode_feature_values_val = []
    if not val_environments:
        log_val_results = False
    val_scores = []
    val_score = -float('inf')

    state_dicts, terminals, all_info = train_environments.reset()

    pbar = tqdm(range(num_steps), unit='Step', disable=not verbose)

    for step in pbar:
        actions = agent.choose_action()  # choose an action for each environment
        next_state_dicts, rewards, terminals, all_info = train_environments.step(actions)
        episode_returns_train = [sum(x) for x in zip(rewards, episode_returns_train)]

        for idx, info in enumerate(all_info):
            episode_feature_values_train[idx].append(info['feature_value'])

        if agent.is_trainable:
            discounts = [discount_factor * (1 - terminal) for terminal in terminals]
            loss = agent.train(state_dicts, actions, next_state_dicts, rewards, discounts, all_info)

            if log_val_results and step % 2000 == 0 or step == num_steps:
                all_feature_values_val = simulate(agent, val_environments,
                                                  num_episodes=10, verbose=False)
                val_score = average_area_under_the_curve(all_feature_values_val)
                val_scores.append(val_score)
                all_episode_feature_values_val.append(all_feature_values_val)

            if loss:
                pbar.set_description('Loss: %0.5f, Val. Score: %0.5f' % (loss, val_score))
            else:  # no loss value is returned inside the burn-in period
                pbar.set_description('Loss: %0.5f, Val. Score: %0.5f' % (float('inf'), val_score))

        state_dicts = next_state_dicts

        for idx, terminal in enumerate(terminals):
            # if terminal then gather episode results for this environment and reset
            if terminal:
                all_episode_returns_train[idx].append(episode_returns_train[idx])
                episode_returns_train[idx] = 0
                all_episode_feature_values_train[idx].append(episode_feature_values_train[idx])
                episode_feature_values_train[idx] = []
                state_dict, terminal, info = train_environments.environments[idx].reset()
                state_dicts[idx] = state_dict

        if step % 2500 == 0 or step == num_steps - 1:  # save model every 2500 steps
            checkpoint_name = 'checkpoint_' + str(step) + '.pt'
            save_path = os.path.join(base_save_path, checkpoint_name)
            save_checkpoint(agent.embedding_module, agent.q_net,
                            agent.optimizer,
                            agent.replay_buffer,
                            all_episode_returns_train, all_episode_feature_values_train,
                            val_scores, all_episode_feature_values_val,
                            step,
                            save_path)

    train_environments.reset()

    train_results = {'returns': all_episode_returns_train,
                     'feature_values_train': all_episode_feature_values_train}

    val_results = {'validation_scores': val_scores,
                   'feature_values_val': all_episode_feature_values_val}

    return train_results, val_results
