from PIL.Image import NONE
import gym
from gym.wrappers import Monitor
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


#################################################################################
# Hyperparameters

BATCH_SIZE = 128
REPLAY_SIZE = 2000
EPISODES = 800
TARGET_MODEL_UPDATE = 200
REPLAY_MEMORY = 50_000

# CNN model params
LEARNING_RATE = 0.00025
KERNEL_SIZE = [8,4,3]
STRIDES = [4,2,1]
POOL_SIZE = 2

# Q-learning params
Q_LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.97

#################################################################################
# Metrics to plot

fit_metrics = ['loss', 'mean_squared_error', 'logcosh', 'cosine_similarity', 'categorical_crossentropy']
fit_history = dict((metric, []) for metric in fit_metrics)
fit_history_ep_avg = dict((metric, []) for metric in fit_metrics)
fit_history_score = []


# Plot training data
def plot():
    for key in fit_metrics:
        plt.plot(fit_history_ep_avg[key])
        plt.title('model '+key)
        plt.ylabel(key)
        plt.xlabel('episode')
        plt.show()
    plt.plot(fit_history_score)
    plt.title('model reward distribution')
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.show()

# update metrics average
def update_history_avg():
    for key in fit_metrics:
        avg = np.average(fit_history[key])
        fit_history_ep_avg[key].append(avg)
        fit_history[key] = []

# update score/reward distribution
def update_history_score(score):
    fit_history_score.append(score)

#################################################################################
# Preprocessing

def preprocess(obs, normalize=False):
    # Crop out score and floor
    img = obs[25:195]  

    # Downsize
    img = img[::2, ::2]

    # Take greyscale (black and white)
    img = img.mean(axis=2)  

    img[img != 0] = 1

    # Is this needed? normalize the image from -1 to +1  
    # No difference visually but tensor is different
    if normalize:
        img = (img - 128) / 128 - 1  
    # reshape to 1D tensor
    return img.reshape(85,80,1)

# frame stacking
# https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
# https://arxiv.org/pdf/1312.5602.pdf
# need to get overlapping sets of frames
# Ex: X1, X2, ... , X7 -> [X1, X2, X3, X4], [X2, X3, X4, X5], ... , [X4, X5, X6, X7]

frame_skip = 4 # only one every four screenshot is considered. If there is no subsampling, not enough information to discern motion
frame_stack_size = 4


def stack_frames(stacked_frames, previous_frame, state, is_new):
    frame = preprocess(state)
    if is_new: # new episode
        # replace stacked_frames with 4 copies of current frame
        for i in range(frame_stack_size):
            stacked_frames.append(frame)
    else:
        # take elementwise maxima of newest frame in stacked_frames and frame
        stacked_frames.append(np.maximum(previous_frame, frame))
    stacked_state = np.stack(stacked_frames)
    return stacked_state, stacked_frames

# Display the preprocessed images
def preprocess_plot():
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
    
    #     initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
    #     initializer = keras.initializers.HeUniform()
    initializer = keras.initializers.GlorotUniform()
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Conv2D(32, kernel_size=KERNEL_SIZE[0], input_shape=(4,85,80,1), activation='relu', 
        padding='same', strides=STRIDES[0], kernel_initializer=initializer))
    #model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))

    # Hidden convolutional layers
    model.add(keras.layers.Conv2D(64, kernel_size=KERNEL_SIZE[1], activation='relu', padding='same', 
        strides=STRIDES[1], kernel_initializer=initializer))
    #model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
    model.add(keras.layers.Conv2D(64, kernel_size=KERNEL_SIZE[2], activation='relu', padding='same', 
        strides=STRIDES[2], kernel_initializer=initializer))
    #model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))

    # Flatten and use fully connected network
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', kernel_initializer=initializer))

    # Output layer
    model.add(keras.layers.Dense(action_shape, activation='softmax'))

    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
        metrics=[keras.metrics.MeanSquaredError(), keras.metrics.LogCoshError(), keras.metrics.CosineSimilarity(), keras.metrics.CategoricalCrossentropy()])

    return model



#################################################################################
# Training agent

def train(env, replay_memory, model, target_model, epoch):

    if len(replay_memory) <= REPLAY_SIZE:
        return

    batch = random.sample(replay_memory, BATCH_SIZE)
    states = np.array([step[0] for step in batch])
    q_values = model.predict(states)
    succesive_states = np.array([step[3] for step in batch])
    succesive_q_values = target_model.predict(succesive_states)

    X_train = []
    Y_train = []
    
    #Decay the q-learning rate (aplha) at each epoch
    alpha = max(0.001, Q_LEARNING_RATE - (Q_LEARNING_RATE - 0.001) * (epoch / 100000))
    
    for i, (state, action, reward, new_state, dead) in enumerate(batch):
        if not dead:
            # Bellman Equation : r(s) + gamma * max_a'(Q(s',a'))
            qValue = reward + DISCOUNT_FACTOR * np.max(succesive_q_values[i])
        else:
            # Pick reward as the episode has ended; no succesive state
            qValue = -1
        
        # TODO: Figure out y_train values work or not
        # Temporal Difference
        # q_value_arr for a state s : [qVal action1, qval action1, ..., qval action18] 
        q_value_arr = q_values[i]
        # Qvalue for action a  : Q(s,a) + alpha(r(s) + gamma*max_a'(Q(s',a')) - Q(s, a))         
        # q_value_arr[action] = (1 - Q_LEARNING_RATE) * q_value_arr[action] + Q_LEARNING_RATE * qValue
        q_value_arr[action] = q_value_arr[action] + alpha * (qValue - q_value_arr[action])

        X_train.append(state)
        Y_train.append(q_value_arr)
    
    if epoch % 250 == 0:
        checkpoint_filepath = "./tmp/cp.ckpt"

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
            save_freq=1,
            monitor='mean_squared_error',
            mode='min')

        np.save('./tmp/fit_history.txt', fit_history)
        np.save('./tmp/fit_history_ep.txt', fit_history_ep_avg)
        np.save('./tmp/fit_history_score.txt', fit_history_score)

        # Model weights are saved if it's the best seen so far.
        history = model.fit(np.array(X_train), np.array(Y_train), batch_size=BATCH_SIZE, callbacks=[model_checkpoint_callback]) 
    else:
        history = model.fit(np.array(X_train), np.array(Y_train), batch_size=BATCH_SIZE)

    for key in history.history.keys():
        fit_history[key].append(history.history[key][0]) # since model.fit with default num epochs = 1

#     # The model weights (that are considered the best) are loaded into the model.
#     model.load_weights(checkpoint_filepath)


#################################################################################
# Deep Q-Learning agent

def DQN_agent(env):

    epsilon = 1
    eps_min = 0.05
    eps_max = 1
    decay = 0.015

    model = network(state_shape, action_shape)
    target_model = network(state_shape, action_shape)
    target_model.set_weights(model.get_weights())

    # initialize with zeroes
    previous_frame = [np.zeros((85,80), dtype=int)]
    stacked_frames = deque(maxlen = frame_stack_size)
    for i in range(frame_stack_size):
        stacked_frames.append([np.zeros((85,80), dtype=int)])

    # Memory buffer to store the last N experiences
    replay_memory = deque(maxlen=REPLAY_MEMORY)

    update_target_counter = 0
    step_counter = 0
    epoch = 0

    for episode in range(EPISODES):
        state = env.reset()
        score = 0 
        done = False
        dead = False
        start_life = 3
        
        state,_,_,_ = env.step(0)
        stacked_state, stacked_frames = stack_frames(stacked_frames, previous_frame, state, True)

        while not done:
            step_counter += 1
            dead = False

            # Epsilon Greedy Strategy with explore probability epsilon
            if np.random.rand() <= epsilon:
                # Explore 
                action = env.action_space.sample()
            else:
                # Exploit best action from cnn
                predictions = model.predict(np.array([stacked_state,])).flatten()
                action = np.argmax(predictions)
            

            new_state, reward, done, info = env.step(action)

            if step_counter % frame_skip == 0:
                new_stacked_state, stacked_frames = stack_frames(stacked_frames, previous_frame, new_state, False)
                
                if start_life > info['lives']:
                    dead = True
                    start_life = info['lives']

                replay_memory.append([stacked_state, action, reward, new_stacked_state, dead])

                stacked_state = new_stacked_state
                state = new_state

            if step_counter % (frame_skip - 1) == 0:
                previous_frame = preprocess(new_state)
            
            score += reward

            if step_counter % (frame_stack_size*frame_skip) == 0 or done:
                    epoch += 1
                    train(env, replay_memory, model, target_model, epoch)


            if update_target_counter >= TARGET_MODEL_UPDATE:
                    update_target_counter = 0
                    target_model.set_weights(model.get_weights())

            if done:
                print('Score: {} after episode = {}'.format(score, episode))
                update_history_avg()
                update_history_score(score)

        # Exponential Decay for epsilon (explore with atleast eps_min probability)
        epsilon = eps_min + (eps_max - eps_min) * np.exp(-decay * episode)
    
    model.save("models/model#")

#################################################################################
# Record test video

def wrap_env_video(env):
  env = Monitor(env, './video', force=True)
  return env


#################################################################################
# Testing the model performance

def test():

    env = wrap_env_video(gym.make('SpaceInvaders-v4', render_mode='human'))

    state = env.reset()

    TEST_EPISODES = 100

    model = keras.models.load_model("models/model#")
    scores = []
    
    # initialize with zeroes
    stacked_frames = deque(maxlen = frame_stack_size)
    for i in range(frame_stack_size):
        stacked_frames.append([np.zeros((85,80), dtype=int)])

    for episode in range(TEST_EPISODES):
        state = env.reset()
        score = 0 
        done = False

        stacked_state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not done:
            predictions = model.predict(np.array([stacked_state,])).flatten()
            action = np.argmax(predictions)


            new_state, reward, done, info = env.step(action)
            new_stacked_state, stacked_frames = stack_frames(stacked_frames, new_state, False)

            score += reward
            stacked_state = new_stacked_state
            state = new_state

            if done:
                scores.append(score)
                print('episode: {}, score: {}'.format(episode, score))

    
    x = np.array([i for i in range(TEST_EPISODES)])
    y = np.array(scores)

    print(np.average(y))
    
    plt.plot(x, y)
    plt.show()


#################################################################################
# Main

if __name__ == "__main__":

    # env = gym.make('SpaceInvaders-v4')
    # env.reset()

    # state_shape = env.observation_space.shape
    # action_shape = env.action_space.n

    # # model = network(state_shape, action_shape)
    # # model.summary()

    # DQN_agent(env)

    # plot()
    # env.close()

    test()

    ### Testing
    # print(env.unwrapped.get_action_meanings())

    # print(state_shape)
    # train()
    # preprocess_plot()

    # print(env.action_space)
    # print(env.observation_space)
    # obs_preprocessed = preprocess(env.env)
    # plt.imshow(obs_preprocessed, cmap='gray')
    # plt.show()


