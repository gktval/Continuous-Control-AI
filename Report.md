## REPORT

### Introduction
This project implements and compares three networks for training agents in Unity's Reacher environment. The three networks tested are DDPG, D4PG, and PPO. 

### 1. DDPG Algorithm
The Deep Deterministic Policy Gradient (DDPG) algorithm is discussed in detail here: https://arxiv.org/abs/1509.02971. To summarize the paper, DDPG uses a 
> model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces

This means that DDPG uses a policy-based algorithm to learn policies, then evaluates the policy with a value forward critical network. Being off-policy, the algorithm can learn from policies that are exploratory. Actor-critic models are two separate functions that model the policy and the action-value spaces for learning. At each timestep, gradient estimations are made by calculating the loss between the two models. 

Below are the hyperparameters for training the DDPG network:
Batch Size = 128
Buffer Size = 1024
Gamma: 0.99
Actor Learning Rate: 0.001
Critic Learning Rate: 0.0005
Noisy Net: Ornstein-Uhlenbeck process

### 2. D4PG Algorithm
The Distributed Distributional Deterministic Policy Gradient (D4PG) algorithm is very similar to the DDPG algorithm with a few exceptions: it uses a prioritized replay buffer, N-step returns, and a distributional critical update. The paper detailing D4PG can be found here: https://openreview.net/forum?id=SyZipzbCb. The paper discusses combining categorical and gaussian operations to predict the critic. In my code, only a categorical operation is achieved on the critic. The atom size was 51, which is used to compute the distributions of the final output from the critic network.

Below are the hyperparameters for training the D4PG network:
Batch Size = 128
Buffer Size = 1024
Gamma: 0.99
Actor Learning Rate: 0.001
Critic Learning Rate: 0.0005
Noisy Net: Ornstein-Uhlenbeck process
Tau = 0.001
N-steps = 5
Priority Epsilon = .0001
N_Atoms = 51
Vmin = -1
Vmax = 1

### 3. Models for DDPG and D4PG
I wanted the comparison between DDPG and D4PG as close as possible for comparisons. Thus, the models were nearly identical between the two. 

Actor 
	Hidden 1: (input, 128) - ReLU
	Hidden 2: (128, 128) - ReLU
	Output: (128, 4) - TanH

Critic
	Hidden 1: (input, 128) - Linear
	Hidden 2: (128, 256) - Linear
	Hidden 3: (256, 128) - Linear
	Hidden 4: (128, 32) - Linear
	Output: (32, 1 [51]) - Linear

In the D4PG model, the output was 51, which is equal to the number of atoms for the distribution update.

### 4. PPO
Proximal Policy Optimization (PPO) was the third algorithm tested. The paper describing PP) can be found here: https://arxiv.org/pdf/1707.06347.pdf. To summarize the PPO algorithm, it is also an actor/critic algorithm (policy and a value) to evaluate continuous control problems. The policy model is used to create trajectories (and advantages) which are then segmented through a number of epochs to construct a surrogate loss. Advantages, which are the normalized losses for each replay in the memory buffer, are created during the rollback on the trajectories. The surrogate losses are clipped (and/or weighted and penalized) during each optimization.

Below are the hyperparameters for training the PPO network:
Batch Size = 256
Buffer Size = 1024
Gamma = 0.99
Actor Learning Rate: 0.001
Epsilon Clip = 0.2
Epochs = 64
Epsilon Optimization = .0005
GAE lambda = 0.9
gradient clip = 0.25
entropy penalty = 0.01
loss weight = 1
    
### 5. PPO model
Shared Network
	Hidden 1: (input, 512) - Linear
	Hidden 2: (512, 512) - ReLU

Actor
	Hidden 1: (512, 512) - Linear
	Output: (512, 4) - TanH

Critic
	Hidden 1: (512, 512) - Linear
	Output: (512, 1) - Linear

### Results
The results of the training will be stored in a folder called `scores` location in the `python` folder. After running several of the deep neural networks, you can replay the trained agent by changing the `isTest` variable passed into the `run()` method. 

Each of the algorithms described above achieved an average score of +30 over 100 episodes as listed below:
DDPG - 116 episodes
D4PG - 193 episodes
PPO - 146 episodes

The PPO algorithm achieved the results in more episodes than DDPG, but the process complete more quickly. Once PPO reach a score of +30, the distribution of rewards was much more compact than DDPG. The D4PG achieved a score of 30 the same number of episodes (35), but it took the D4PG algorithm much longer to reach the average score of 30 over 100 episodes because it plateaued and did not achieve as high of scores as DDPG or PPO.


The plot of rewards can be found here:
https://github.com/gktval/DQN-ContinuousControl/blob/main/results.png
The scores from each agent can be found here:
https://github.com/gktval/ContinuousControl/blob/main/python/scores/
The replay from each agent can be found here:
https://github.com/gktval/ContinuousControl/blob/main/python/checkpoints/

### Future work
Future work should include should compare the AC3 network, and other continuous control networks not presented in this project. Further research could explore fine tuning of each of the parameters used in the DDPG, D4PG and PPO algorithms. Furthermore, as presented in the D4PG research article, a gaussian distribution could be used to improve the loss. A lot of work is needed to improve the fine tuning of the model and the efficiency of the D4PG model. Since the model took so long to run, the hyperparameters were not optimized.

A better approach than comparing the three algorithms based on episodes would be to compare the time it took to achieve the score of 30+ for 100 episodes. 


