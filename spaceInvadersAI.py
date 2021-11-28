import gym
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
env.reset()

state_shape = env.observation_space.shape
action_shape = env.action_space.n

#################################################################################
# Preprocessing

def preprocess(obs, normalize=False):
    # Crop out score and floor
    img = obs[25:195]  

    # Downsize
    img = img[::2, ::2]

    # Take greyscale (black and white)
    img = img.mean(axis=2)  

    # color = np.array([210, 164, 74]).mean()
    # img[img==color] = 0  
    # img[img==144] = 0
    # img[img==109] = 0
    img[img != 0] = 1

    # Is this needed? normalize the image from -1 to +1  
    # No difference visually but tensor is different
    if normalize:
        img = (img - 128) / 128 - 1  

    print("before: ", obs.shape)
    print("after: ", img.shape)

    # reshape to 1D tensor
    return img.reshape(85,80)

# frame stacking
# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
# https://arxiv.org/pdf/1312.5602.pdf
# need to get overlapping sets of frames
# Ex: X1, X2, ... , X7 -> [X1, X2, X3, X4], [X2, X3, X4, X5], ... , [X4, X5, X6, X7]

frame_skip = 4 # only one every four screenshot is considered. If there is no subsampling, not enough information to discern motion
frame_stack_size = 4

# initialize with zeroes
stacked_frames = deque(maxlen = frame_stack_size)
for i in range(frame_stack_size):
    stacked_frames.append([np.zeros((85,80), dtype=int)])


def stack_frames(stacked_frames, state, is_new):
    frame = preprocess(state)
    if is_new: # new episode
        # replace stacked_frames with 4 copies of current frame
        for i in range(frame_stack_size):
            stacked_frames.append(frame)
    else:
        # take elementwise maxima of newest frame in stacked_frames and frame
        stacked_frames.append(np.maximum(stacked_frames[3],frame))
    stacked_state = np.stack(stacked_frames)
    return stacked_state, stacked_frames

# Display the preprocessed images
def preproces_plot():
    obs = env.reset()

    for i in range(10):
        # if i > 20:
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(obs)
        f.add_subplot(1,2,2)
        plt.imshow(preprocess(obs), cmap='gray')
        # plt.show(block=True)

        # env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

#################################################################################
# Model

def agent(state_shape, action_shape):

    learning_rate = 0.001
    
    initializer = keras.intialiazers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', 
        kernel_initializer=initializer))
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

#################################################################################
# Training agent

def train():
    episodes = 10
    for episode in range(episodes):
        env.reset()
        score = 0 
        done = False

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))

#################################################################################
# Main

if __name__ == "__main__":
    # print(env.unwrapped.get_action_meanings())

    # print(state_shape)
    # train()
    # preproces_plot()


    # print(env.action_space)
    # print(env.observation_space)
    # obs_preprocessed = preprocess(env.env)
    # plt.imshow(obs_preprocessed, cmap='gray')
    # plt.show()
    env.close()