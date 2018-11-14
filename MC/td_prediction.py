import sys
from collections import defaultdict

import matplotlib

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')
env = BlackjackEnv()


def td_prediction(policy, env, num_episodes, discount_factor=1.0, alpha=0.1):
    # The final value function
    V = defaultdict(float)

    # Implement this!
    for i_episode in range(1, 1 + num_episodes):
        if i_episode % 10000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # Following the policy to play the game and record states
        observation = env.reset()
        is_done = False
        while not is_done:
            action = policy(observation)
            new_observation, reward, is_done, _ = env.step(action)
            # update V
            if not is_done:
                estimate_next_v = V[new_observation]
            else:
                estimate_next_v = 0
            td_target = reward + discount_factor * estimate_next_v
            td_delta = td_target - V[observation]
            V[observation] += alpha * td_delta
            observation = new_observation
    return V


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    # Implement this!
    for i_episode in range(1, 1 + num_episodes):
        if i_episode % 10000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # Following the policy to play the game and record states
        episode_states = []
        observation = env.reset()
        is_done = False
        while not is_done:
            action = policy(observation)
            new_observation, reward, is_done, _ = env.step(action)
            episode_states.append((observation, reward, is_done))
            observation = new_observation

        # Evaluating policy (updating value function)
        for state in set(tuple(x[0]) for x in episode_states):
            first_visit_of_state = next(i for i, x in enumerate(episode_states) if x[0] == state)
            g = sum(x[1] * discount_factor ** i for i, x in enumerate(episode_states[first_visit_of_state:]))
            returns_count[state] += 1
            returns_sum[state] += g

            V[state] = returns_sum[state] / returns_count[state]
    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 17 else 1


V_500k = td_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="td_prediction  500,000 Steps")
