# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.7
# ---

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

action_dim = 3
sample_N = 100
Q_vs = np.random.rand(sample_N, action_dim)
Q_values = tf.constant(Q_vs, dtype=tf.float32)
Q_values.shape

Q_max_values = tf.reduce_max(Q_values, axis=-1)
with tf.Session() as sess:
    print(Q_max_values.eval().shape)

action_indexes = np.random.randint(action_dim, size=sample_N)
action_indexes.shape

indices_d = np.dstack((np.arange(sample_N), action_indexes))
# indices

# np.vstack([np.arange(sample_N), action_indexes])
indices = np.array([[i, action] for i, action in enumerate(action_indexes)])
# indices

assert np.all(indices_d == indices)
assert np.all(indices_d == np.stack((np.arange(sample_N), action_indexes), axis=1))

with tf.Session() as sess:
    assert np.all(tf.stack((tf.range(sample_N), action_indexes), axis=-1).eval() == indices)

with tf.Session() as sess:
    print(tf.gather_nd(Q_values, indices_d).eval().shape)

done_mask_ph = np.random.rand(1, sample_N)
done_mask_ph = done_mask_ph > 0.5
done_mask_ph = done_mask_ph.astype(int)[0]
done_mask_ph

inputs = np.array(np.arange(action_dim * sample_N).reshape(-1, action_dim))
inputs.shape

with tf.Session() as sess:
    for axis_index, shape in enumerate(inputs.shape):
        print(axis_index, shape)
        if axis_index == 1:
            print("axis=%s ==> %s" % (axis_index, tf.gather(inputs, [1], axis=axis_index).eval().shape))
        else:
            print("axis=%s ==> %s" % (axis_index, tf.gather(inputs, [0, 2], axis=axis_index).eval().shape))

# using tf.one_hot for Q_values filter
with tf.Session() as sess:
    print(tf.one_hot(action_indexes, action_dim).eval().shape)
#     print(tf.one_hot(action_indexes, action_dim) * Q_values)
    print(tf.multiply(tf.one_hot(action_indexes, action_dim), Q_values).eval().shape)
#     as tf.one_hot will remove all actions with zero values
    print(tf.reduce_sum(tf.one_hot(action_indexes, action_dim) * Q_values, axis=-1).eval().shape)
    print(tf.reduce_sum(tf.one_hot(action_indexes, action_dim) * Q_values, axis=0).eval().shape)

# # Q-Network Analysis

# +
import gym
from gym import wrappers
import random
import os.path as osp
from collections import namedtuple

from atari_wrappers import *
from dqn_utils import *

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def wrap_deepmind(env):
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ClippedRewardsWrapper(env)
    return env

# +
env = gym.make('PongNoFrameskip-v4')

seed = random.randint(0, 9999)
set_global_seeds(seed)
env.seed(seed)

expt_dir = '/tmp/hw3_vid_dir2/'
env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
env = wrap_deepmind(env)

# +
replay_buffer_size=1000000
frame_history_len=4
lander=False
gamma=0.99
grad_norm_clipping=10
num_timesteps=2e8
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

# This is just a rough estimate
num_iterations = float(num_timesteps) / 4.0

lr_multiplier = 1.0
lr_schedule = PiecewiseSchedule([(0,                   1e-4 * lr_multiplier),
                                 (num_iterations / 10, 1e-4 * lr_multiplier),
                                 (num_iterations / 2,  5e-5 * lr_multiplier),],
                                outside_value=5e-5 * lr_multiplier)
optimizer_spec = OptimizerSpec(
    constructor=tf.train.AdamOptimizer,
    kwargs=dict(epsilon=1e-4),
    lr_schedule=lr_schedule
)

def stopping_criterion(env, t):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

exploration_schedule = PiecewiseSchedule(
    [
        (0, 1.0),
        (1e6, 0.1),
        (num_iterations / 2, 0.01),
    ], outside_value=0.01
)
# -

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

q_func = atari_model

# +
if len(env.observation_space.shape) == 1:
    # This means we are running on low-dimensional observations (e.g. RAM)
    input_shape = env.observation_space.shape
else:
    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)

num_actions = env.action_space.n
obs_t_ph = tf.placeholder(tf.float32 if lander else tf.uint8, [None] + list(input_shape))
# placeholder for current action
act_t_ph = tf.placeholder(tf.int32,   [None])
# placeholder for current reward
rew_t_ph = tf.placeholder(tf.float32, [None])
# placeholder for next observation (or state)
obs_tp1_ph = tf.placeholder(tf.float32 if lander else tf.uint8, [None] + list(input_shape))
done_mask_ph = tf.placeholder(tf.float32, [None])

# casting to float on GPU ensures lower data transfer times.
if lander:
    obs_t_float = obs_t_ph
    obs_tp1_float = obs_tp1_ph
else:
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

# YOUR CODE HERE - Problem 1.3 Implementation 
## 1. q_values network
q_allaction_values = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
q_action_values = tf.reduce_sum(q_allaction_values * tf.one_hot(act_t_ph, num_actions), axis=-1)
q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

## 2. target_q_value network
q_prime_values = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)
target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')
y_values = rew_t_ph + gamma * (1 - done_mask_ph) * tf.reduce_max(q_prime_values, axis=-1)

# for q_value and q_target_value's Bellman error
total_error = huber_loss(q_action_values - y_values)

######

# construct optimization op (with gradient clipping)
learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
train_fn = minimize_and_clip(optimizer, total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)

# update_target_fn will be called periodically to copy Q network to target Q network
update_target_fn = []
for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                           sorted(target_q_func_vars, key=lambda v: v.name)):
    update_target_fn.append(var_target.assign(var))
update_target_fn = tf.group(*update_target_fn)

# construct the replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
replay_buffer_idx = None

###############
# RUN ENV     #
###############
model_initialized = False
num_param_updates = 0
mean_episode_reward      = -float('nan')
best_mean_episode_reward = -float('inf')
last_obs = env.reset()
log_every_n_steps = 10000

start_time = None
t = 0

# +
last_obs = env.reset()
replay_buffer_idx = replay_buffer.store_frame(last_obs)
obs_Q_input = replay_buffer.encode_recent_observation()

print("original obs_Q_input.shape = ", obs_Q_input.shape)
# obs_Q_input = obs_Q_input.reshape([-1] + list(obs_Q_input.shape))
# print("Q network input obs_Q_input.shape = ", obs_Q_input.shape)
# -

with tf.Session() as sess:
    print(lr_schedule.value(t))
#     q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
#     initialize_interdependent_variables(sess, q_func_vars)
#     print(sess.run(tf.argmax(q_allaction_values, axis=-1), feed_dict={obs_t_float:obs_Q_input[None]}))

# # PiecewiseSchedule 
# estimate value by current step number

lr_schedule = PiecewiseSchedule([
                                 (0,                   1e-4 * lr_multiplier),
                                 (num_iterations / 10, 1e-4 * lr_multiplier),
                                 (num_iterations / 4,  5e-5 * lr_multiplier),
                                 (num_iterations / 2,  5e-5 * lr_multiplier),
                                ],
                                outside_value=5e-5 * lr_multiplier)
print(list(zip(lr_schedule._endpoints[:-1], lr_schedule._endpoints[1:])))
