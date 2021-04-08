import sys

sys.path.append('/usr/local/lib/python3.9/site-packages')
import gym
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial

plt.style.use('ggplot')
env = gym.make('Blackjack-v0')


# Define a policy where we hit until we reach 19
# Actions here are 0-stand, 1-hit

# observation encompasses all the data about the state
def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 19 else 1


# define method to generate according to pur policy
def generate_episode(policy, env):
    states, actions, rewards = [], [], []

    # Initialize the gym environment
    observation = env.reset()
    while True:
        # append state to states list
        states.append(observation)

        # select action according to our policy, append the action to actions list
        action = policy(observation)
        actions.append(action)

        # perform the action on env and observe the reward, append the reward to rewards list
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        # break if the state is terminal state
        if done:
            break
    return states, actions, rewards


# define first visit monte carlo prediction method
def first_visit_mc_prediction(policy, env, n_episodes):
    # Initializing the empty value table as dictionary for storing state values
    value_table = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        # generate episode according to our policy
        states, _, rewards = generate_episode(policy, env)
        returns = 0
        for t in range(len(states) - 1, 0, -1):
            R = rewards[t]
            S = states[t]
            returns += R

            # check if episode is visited first time, then
            # NEW_ESTIMATE = OLD_ESTIMATE + STEP_SIZE(TARGET - OLD_ESTIMATE)
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - value_table[S]) / N[S]
    return value_table


def plot_blackjack(V, ax1, ax2):
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    usable_ace = np.array([False, True])
    state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))
    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]
    X, Y = np.meshgrid(player_sum, dealer_show)
    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])
    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player-sum')
        ax.set_xlabel('dealer-sum')
        ax.set_zlabel('state-value')
    plt.show()


value = first_visit_mc_prediction(sample_policy, env, n_episodes=1000000)

fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8), subplot_kw={'projection': '3d'})
axes[0].set_title('state-value distribution w/o usable ace')
axes[1].set_title('state-value distribution w/ usable ace')
plot_blackjack(value, axes[0], axes[1])
