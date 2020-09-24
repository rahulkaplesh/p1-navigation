from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import os
import pickle
import logging

env = UnityEnvironment(file_name="C:\Development\DEEPRL\p1-navigation\VisualBanana_Windows_x86_64\Banana.exe")

from agent import Agent

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.visual_observations[0]
state = np.moveaxis(state, -1, 1)
stacked_state = np.array([np.vstack((state,state,state,state))]) ## I will be stacking 4 frames together to make a video of last 3 frames plus a move ! Double DQN Implementation
stacked_state = np.moveaxis(stacked_state, 2, 1)
state_size = stacked_state.shape

agent = Agent(state_size = state_size, action_size = action_size, seed = 0)

scores_window = deque(maxlen=100)
scores = []
if (os.path.exists('scores.data')):
    with open('scores.data', 'rb') as filehandle:
    # read the data as binary data stream
        scores = pickle.load(filehandle)
    with open('epsilon.data', 'rb') as filehandle:
        eps_start = pickle.load(filehandle)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint_local_model.pth'))
    agent.qnetwork_target.load_state_dict(torch.load('checkpoint_target_model.pth'))
    for score in scores[-100:]:
        scores_window.append(score)
else:
    eps_start = 1.0

#summary(agent.qnetwork_local,(state_size,))

start_ep = len(scores) + 1
n_episodes = start_ep + 100
max_steps_per_ep = 1000
eps_end = 0.01
eps_decay = 0.995

logger = logging.getLogger(__name__)

f_handle = logging.FileHandler(str(start_ep) + "_" + str(n_episodes) + "_" + "Log_File.txt")
f_format = logging.Formatter('%(levelname)s: %(asctime)s %(message)s')
f_handle.setFormatter(f_format)
f_handle.setLevel(logging.INFO)

logger.addHandler(f_handle)


eps = eps_start

for ep in range(start_ep,n_episodes):
    stacked_state = []
    env_info = env.reset(train_mode=True)[brain_name]
    state_obs = env_info.visual_observations[0]
    state1 = np.moveaxis(state_obs, -1, 1)
    env.step(np.random.randint(action_size))
    state_obs = env_info.visual_observations[0]
    state2 = np.moveaxis(state_obs, -1, 1)
    env.step(np.random.randint(action_size))
    state_obs = env_info.visual_observations[0]
    state3 = np.moveaxis(state_obs, -1, 1)
    env.step(np.random.randint(action_size))
    state_obs = env_info.visual_observations[0]
    state4 = np.moveaxis(state_obs, -1, 1)
    stacked_state = np.vstack((state1,state2,state3,state4))
    stacked_state = np.moveaxis(stacked_state, 0, 1)
    #print(stacked_state.shape)
    score = 0
    for step in range(max_steps_per_ep):
        action = agent.act(stacked_state, eps)
        env_info = env.step(action.astype(int))[brain_name]
        next_state = env_info.visual_observations[0]
        next_state = np.moveaxis(next_state, -1, 1)
        processing_state = np.moveaxis(stacked_state, 1, 0)
        new_stacked_state = np.vstack((processing_state[1:2,:,:,:],processing_state[2:3,:,:,:],processing_state[3:4,:,:,:],next_state))
        new_stacked_state = np.moveaxis(new_stacked_state, 0, 1)
        #print(new_stacked_state.shape)        
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(stacked_state,action,reward,new_stacked_state,done)
        stacked_state = new_stacked_state
        score += reward
        if done:
            break
    eps = max(eps_end, eps_decay*eps)
    scores_window.append(score)
    scores.append(score)
    logger.info('Episode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_window)))
    #print('\rEpisode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_window)), end="")
    if np.mean(scores_window)>=12.0:
        logger.info('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep, np.mean(scores_window)))
        #print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep-100, np.mean(scores_window)))
        break

torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_local_model.pth')
torch.save(agent.qnetwork_target.state_dict(), 'checkpoint_target_model.pth')

with open('scores.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(scores, filehandle)

with open('epsilon.data', 'wb') as filehandle:
    pickle.dump(eps, filehandle)

env.close()