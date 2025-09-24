import gymnasium as gym



#Random Moves no Q-learning implemented
def run():
    
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")

    state = env.reset()[0]   #States: 0 to 63
    terminated = False       # True when fall in hole or reach goal
    truncated = False        # True when episode duration exceeds max length


    while(not terminated and not truncated):

        action = env.action_space.sample()
        new_state, reward, terminated, truncated,_ = env.step(action)
        state = new_state

    env.close()

if __name__ == "__main__":
    run()
