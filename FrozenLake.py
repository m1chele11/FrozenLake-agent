import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pickle



#Random Moves no Q-learning implemented
def run(episodes, render=False):

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human" if render else None)

    q_table = np.zeros((env.observation_space.n, env.action_space.n)) # 64 states, 4 actions


    learning_rate = 0.9 # Alpha or learning rate
    discount_factor = 0.9 # Gamma or discount factor

    epsilon = 1 # Exploration rate
    epsilon_decay = 0.0001 # Decay rate for exploration. 1 / .0001 = 10000 episodes to decay to 0
    rng = np.random.default_rng() # Random number generator


    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]   #States: 0 to 63
        terminated = False       # True when fall in hole or reach goal
        truncated = False        # True when episode duration exceeds max length


        while(not terminated and not truncated):
            if(rng.random() > epsilon): # Exploit learned values
                action = np.argmax(q_table[state]) # Choose action with highest Q-value for current state
            else: # Explore
                action = env.action_space.sample()

            new_state, reward, terminated, truncated,_ = env.step(action)

            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action]
            )
            
            state = new_state
    
        epsilon = max(epsilon - epsilon_decay, 0) # Decay epsilon


        if(epsilon ==0):
            learning_rate = .0001

        if(reward == 1):
            rewards_per_episode[i] = 1

    env.close()


    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(sum_rewards)
    plt.savefig("frozenlake.png")

    f = open("frozenlake.pkl", "wb")
    pickle.dump(q_table, f)
    f.close()

if __name__ == "__main__":
    run(15000)
