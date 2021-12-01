import gym
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from itertools import islice


#################################################################################
# Hyperparameters

BATCH_SIZE = 64
REPLAY_SIZE = 1000
EPISODES = 300
TARGET_MODEL_UPDATE = 100
REPLAY_MEMORY = 100_000

# CNN model params
LEARNING_RATE = 0.001
KERNEL_SIZE = 3
POOL_SIZE = 2

# Q-learning params
Q_LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.5


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
    return img.reshape(85,80,1)

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
        plt.show(block=True)

        # env.render()
        # action = env.action_space.sample()
        # obs, reward, done, info = env.step(action)



#################################################################################
# Model - Create a convolutional neural network with Keras

def network(state_shape, action_shape):
    
    initializer = keras.initializers.HeUniform()
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Conv2D(32, kernel_size=KERNEL_SIZE, input_shape=state_shape, activation='relu', 
        padding='same', kernel_initializer=initializer))
    model.add(keras.layers.AveragePooling2D(pool_size=POOL_SIZE))

    # Hidden convolutional layers
    model.add(keras.layers.Conv2D(64, kernel_size=KERNEL_SIZE, input_shape=state_shape, activation='relu', 
        padding='same', kernel_initializer=initializer))
    model.add(keras.layers.AveragePooling2D(pool_size=POOL_SIZE))
    model.add(keras.layers.Conv2D(64, kernel_size=KERNEL_SIZE, input_shape=state_shape, activation='relu', 
        padding='same', kernel_initializer=initializer))
    model.add(keras.layers.AveragePooling2D(pool_size=POOL_SIZE))

    # Flatten and use fully connected network
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu', kernel_initializer=initializer))

    # Output layer
    model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=initializer))

    model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    
    return model



#################################################################################
# Training agent

def train(env, replay_memory, model, target_model):

    replay_len = len(replay_memory)

    if len(replay_memory) <= REPLAY_SIZE:
        return

    batch = random.sample(replay_memory, BATCH_SIZE)
    states = np.array(step[0] for step in batch)
    q_values = model.predict(states)
    succesive_states = np.array(step[3] for step in batch)
    succesive_q_values = target_model.predict(succesive_states)

    X_train = []
    Y_train = []

    for i, (state, action, reward, new_state, done) in enumerate(batch):
        if not done:
            # Bellman Equation : r(s) + gamma * max_a'(Q(s',a'))
            qValue = reward + DISCOUNT_FACTOR * np.max(succesive_q_values[i])
        else:
            # Pick reward as the episode has ended; no succesive state
            qValue = reward
        
        # TODO: Figure out y_train values work or not
        # Temporal Difference
        # q_value_arr for a state s : [qVal action1, qval action1, ..., qval action18] 
        q_value_arr = q_values[i]
        # Qvalue for action a  : Q(s,a) + alpha(r(s) + gamma*max_a'(Q(s',a')) - Q(s, a))         
        q_value_arr[action] = q_value_arr[action] + LEARNING_RATE(qValue - q_value_arr[action])

        X_train.append(state)
        Y_train.append()
    
    model.fit(np.array(X_train), np.array[Y_train], batch_size=BATCH_SIZE)



#################################################################################
# Deep Q-Learning agent

def DQN_agent(env):
    epsilon = 1
    decay = 0.01

    model = network(state_shape, action_shape)
    target_model = network(state_shape, action_shape)
    target_model.set_weights(model.get_weights())

    # Memory buffer to store the last N experiences
    replay_memory = deque(maxlen=REPLAY_MEMORY)

    update_target_counter = 0
    step_counter = 0

    for episode in range(EPISODES):
        state = env.reset()
        score = 0 
        done = False

        while not done:
            step_counter += 1

            # Epsilon Greedy Strategy with explore probability epsilon
            if np.random.rand() <= epsilon:
                # Explore 
                action = env.action_space.sample()
            else:
                # Exploit best action from cnn
                preprocessed_state = preprocess(state) # Preprocess state
                predictions = model.predict(preprocessed_state).flatten()
                action = np.argmax(predictions)
            
            new_state, reward, done, info = env.step(action)
            replay_memory.append([preprocess(state), action, reward, preprocess(new_state), done]) # ERROR CHECK: When done cannot preprocess next state

            if step_counter % 4 == 0 or done:
                train(env, replay_memory, model, target_model)

            score += reward
            state = new_state

            if update_target_counter >= TARGET_MODEL_UPDATE:
                    update_target_counter = 0
                    target_model.set_weights(model.get_weights())

            if done:
                  print('Score: {} after epsidoe = {} and final reward = {}'.format(score, episode, reward))
            
        # Exponetial decay for epsilon (explore with atleast 0.01 or 1% probability)
        epsilon = 0.01 + 0.99 * np.exp(-decay*episode)



#################################################################################
# Main

if __name__ == "__main__":

    env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    env.reset()

    state_shape = env.observation_space.shape
    action_shape = env.action_space.n

    model = network(state_shape, action_shape)
    model.summary()

    env.close()

    ### Testing
    # print(env.unwrapped.get_action_meanings())

    # print(state_shape)
    # train()
    # preproces_plot()

    # print(env.action_space)
    # print(env.observation_space)
    # obs_preprocessed = preprocess(env.env)
    # plt.imshow(obs_preprocessed, cmap='gray')
    # plt.show()