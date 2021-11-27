import gym
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

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


env = gym.make('ALE/SpaceInvaders-v5')
obs = env.reset()


height, width, channels = env.observation_space.shape
actions = env.action_space.n
print(env.unwrapped.get_action_meanings())

for i in range(10):
    # if i > 20:
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(obs)
    f.add_subplot(1,2,2)
    plt.imshow(preprocess(obs), cmap='gray')
    # plt.show(block=True)

    env.render()

    obs, _, _, _ = env.step(env.action_space.sample())



env.close()



# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
    
#     while not done:
#         env.render()
#         action = random.choice([0,1,2,3,4,5])
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# print(type(obs))


# print(env.action_space)
# print(env.observation_space)
# obs_preprocessed = preprocess(env.env)
# plt.imshow(obs_preprocessed, cmap='gray')
# plt.show()