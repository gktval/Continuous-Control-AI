import torch
from d4pg_agent import Agent
from collections import deque
import numpy as np
import os

def Run(env, config, n_episodes=800, max_t=1000):   

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.GAMMA = 0.99
    config.LR_Actor = 1e-3
    config.LR_Critic = 5e-4
    config.PRIORITY_EPS = 1e-4
    config.TAU = 1e-3 
    config.seed = 0
    config.USE_NOISY_NETS = True
    config.USE_PRIORITY_REPLAY = True
    config.N_STEPS = 5
    
    agent = Agent(state_size=state_size, action_size=action_size, config=config)
    scores_window = deque(maxlen=100)  # last 100 scores
    total_scores = []
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations            # get the current state (for each agent)
        scores = 0                                      # initialize the score (for each agent)
        agent.reset()                                   # reset noise
        
        for _ in range(max_t):
            actions = agent.act(states)              

            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                 # see if episode has finished
            scores += np.array(rewards)                       # update the score (for each agent)

            agent.step(states, actions, rewards, next_states, dones)       # step through agent learning 

            states = next_states                      # update the next states

            if np.any(dones):                                  # exit loop if episode finished
                break

        scores_window.append(np.mean(scores))       # save most recent score
        total_scores.append(np.mean(scores))       # update the total scores
        avg_score = np.mean(scores_window)

        print('\rEpisode {}\tCurrent Score: {:.2f}\tAverage Score: {:.2f} '
              .format(i_episode, np.mean(scores), avg_score), end="")

        agent.lr_actor *= config.LR_rate_decay
        agent.lr_critic *= config.LR_rate_decay
        agent.set_learning_rate(agent.lr_actor, agent.lr_critic)
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))

        if avg_score >= 30.0:
            saveCheckpoints(str(config.model.name), agent, config)
            break

    env.close()
    return total_scores, scores_window

def saveCheckpoints(filename, agent, config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    torch.save(agent.actor_local.state_dict(), config.model_path + filename + '_checkpoint-actor.pth')
    torch.save(agent.critic_local.state_dict(), config.model_path + filename + '_checkpoint-critic.pth')

def watchAgent(config, env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.GAMMA = 0.99
    config.LR_Actor = 1e-3
    config.LR_Critic = 2e-3
    config.TAU = 1e-3 
    config.seed = 0
    config.USE_NOISY_NETS = True
    config.USE_PRIORITY_REPLAY = True
    config.N_STEPS = 5
    
    agent = Agent(state_size=state_size, action_size=action_size, config=config)
    agent.actor_local.load_state_dict(torch.load("checkpoints/d4pg_checkpoint-actor.pth"))
    agent.critic_local.load_state_dict(torch.load("checkpoints/d4pg_checkpoint-critic.pth"))
    
    for j in range(10000):
        action = agent.act(states)
                
        env_info = env.step(action)[brain_name]        # send the action to the environment
        done = env_info.local_done[0]                  # see if episode has finished
        if done:
            break 
            
    env.close()