import os
from copy import deepcopy
from tqdm import tqdm
from GraphRL.helpers_miscellaneous import average_area_under_the_curve
from GraphRL.helpers_miscellaneous import save_checkpoint


def simulate(agent, environments, num_episodes=100, verbose=True):
    """
    Simulate agent in multiple environments for a specified number of episodes.
    This function assumes that episodes in each environment have the same number
    of steps.
    """

    agent = deepcopy(agent)
    agent.environments = environments

    all_feature_values = [[] for _ in range(environments.num_environments)]

    state_dicts, terminals, all_info = environments.reset()

    pbar = tqdm(range(num_episodes), unit='Episode', disable=not verbose)

    for _ in pbar:
        episode_feature_values = [[] for _ in range(environments.num_environments)]

        while not any(terminals):
            assert any(terminals) == all(terminals), "Simulation error!"
            actions = agent.choose_action()  # choose an action for each environment
            state_dicts, rewards, terminals, all_info = environments.step(actions)
            episode_feature_values.append(all_info)

            for idx, info in enumerate(all_info):
                episode_feature_values[idx].append(info['feature_value'])

        for idx, terminal in enumerate(terminals):
            if terminal:  # this should always be true when control is here
                all_feature_values[idx].append(episode_feature_values[idx])

        state_dicts, terminals, all_info = environments.reset()

    return all_feature_values


def learn_environments(agent, train_environments, val_environments,
                       num_steps, discount_factor, val_every,
                       save_folder=None,
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
    val_steps = []
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

            if log_val_results and step % val_every == 0:  # evaluate validation performance
                all_feature_values_val = simulate(agent, val_environments,
                                                  num_episodes=10, verbose=False)
                val_score = average_area_under_the_curve(all_feature_values_val)
                val_steps.append(step)
                val_scores.append(val_score)
                all_episode_feature_values_val.append(all_feature_values_val)

                if save_folder is not None:  # save checkpoint
                    save_path = os.path.join(save_folder, 'model_' + str(step) + '.pt')
                    save_checkpoint(agent.embedding_module, agent.q_net,
                                    val_steps, val_scores,
                                    step,
                                    save_path)

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

    train_environments.reset()

    train_log = {'returns': all_episode_returns_train,
                 'feature_values_train': all_episode_feature_values_train}

    # validate final model and save checkpoint
    all_feature_values_val = simulate(agent, val_environments,
                                      num_episodes=10, verbose=False)
    val_score = average_area_under_the_curve(all_feature_values_val)
    val_steps.append(num_steps)
    val_scores.append(val_score)
    all_episode_feature_values_val.append(all_feature_values_val)

    if save_folder is not None:  # save checkpoint
        save_path = os.path.join(save_folder, 'model_' + str(num_steps) + '.pt')
        save_checkpoint(agent.embedding_module, agent.q_net,
                        agent.optimizer,
                        agent.replay_buffer,
                        all_episode_returns_train, all_episode_feature_values_train,
                        val_steps, val_scores, all_episode_feature_values_val,
                        num_steps,
                        save_path)

    val_log = {'validation_steps': val_steps,
               'validation_scores': val_scores,
               'feature_values_val': all_episode_feature_values_val}

    return train_log, val_log
