#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import numpy as np
import tensorflow as tf

import load_policy
import tf_util


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    def _expert_data_gen_fn(num_rollouts=1, verbose=False):
        with tf.Session():
            tf_util.initialize()

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(num_rollouts):
                if verbose:
                    print('Generated expert data: iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = policy_fn(obs[None, :])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("rollout: %i - %i/%i" % (i, steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            # print('returns', returns)
            # print('mean return', np.mean(returns))
            # print('std of return', np.std(returns))

            # Flatten dimensions for niceness
            observations = np.array(observations)
            actions = np.array(actions)

            return {
                'observations': observations.reshape(observations.shape[0], -1),
                'actions': actions.reshape(actions.shape[0], -1)
            }

    clone(_expert_data_gen_fn)


def clone(expert_data_gen_fn):

    # Parameters
    learning_rate = 0.01
    training_iters = 200000
    display_iter = 1
    rollouts = 1

    n_hidden = 1024

    expert_data = expert_data_gen_fn(num_rollouts=rollouts)
    obs_shape = np.prod(expert_data['observations'][0].shape)
    action_shape = np.prod(expert_data['actions'][0].shape)
    batch_size = 500

    # Network Parameters
    n_input = obs_shape
    n_action_dims = action_shape
    # dropout = 0.75 # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_action_dims])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # Create model
    def fc_net(x, weights, biases, dropout):

        # Fully connected layer
        fc1 = tf.matmul(x, weights['wd1']) + biases['bd1']

        # Nonlinearity
        fc1 = tf.nn.relu(fc1)

        # Apply Dropout
        # fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.matmul(fc1, weights['out']) + biases['out']
        return out

    # Store layers weight & bias
    weights = {
        # fully connected, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([n_input, n_hidden]) * 0.01),
        # n_hidden inputs, 10 outputs
        'out': tf.Variable(tf.random_normal([n_hidden, n_action_dims]) * 0.01)
    }

    biases = {
        'bd1': tf.Variable(tf.zeros([n_hidden])),
        'out': tf.Variable(tf.zeros([n_action_dims]))
    }

    # Construct model
    pred = fc_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.l2_loss(pred -y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # def minibatch(expert_observations, expert_actions, batch_size, batch_idx):
    #     batches_per_epoch = len(expert_observations) / batch_size
    #     obs_batch = []
    #     action_batch = []
    #     mod_batch_idx = batch_idx % batches_per_epoch
    #     for i, (obs, action) in enumerate(zip(expert_observations, expert_actions)):
    #         if i % batches_per_epoch == mod_batch_idx:
    #             obs_batch.append(obs)
    #             action_batch.append(action)
    #
    #     return np.array(obs_batch), np.array(action_batch)


    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        training_iter_num = 1
        # Keep training until reach max iterations
        while training_iter_num < training_iters:

            # batch_x, batch_y = minibatch(expert_data['observations'],
            #                              expert_data['actions'],
            #                              batch_size,
            #                              training_iter_num)

            batch_x = expert_data['observations']
            batch_y = expert_data['actions']

            # Run optimization op (backprop)
            # sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if training_iter_num % display_iter == 0:
                # Calculate batch loss and accuracy
                loss = sess.run([cost], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(training_iter_num) + ", Minibatch Loss= {}".format(loss))
            training_iter_num += 1

        print("Optimization Finished!")

        # Calculate accuracy
        test_expert_data = expert_data_gen_fn()
        print("Testing Accuracy: TODO")

if __name__ == '__main__':
    main()
