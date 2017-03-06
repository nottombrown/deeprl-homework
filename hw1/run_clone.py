#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import os

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
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
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

    expert_data_minibatch = expert_data_gen_fn()
    obs_shape = np.prod(expert_data_minibatch['observations'][0].shape)
    action_shape = np.prod(expert_data_minibatch['actions'][0].shape)

    # Parameters
    learning_rate = 0.01
    training_iters = 200000
    display_iter = 10
    rollouts_per_batch = 100

    n_hidden = 1024

    # Network Parameters
    n_input = obs_shape
    n_classes = action_shape # MNIST total classes (0-9 digits)
    dropout = 0.75 # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


    # Create model
    def fc_net(x, weights, biases, dropout):

        # Fully connected layer
        fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # fully connected, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        # n_hidden inputs, 10 outputs
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }

    biases = {
        'bd1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = fc_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        training_iter_num = 1
        # Keep training until reach max iterations
        while training_iter_num < training_iters:
            expert_data_minibatch = expert_data_gen_fn(num_rollouts=rollouts_per_batch)
            batch_x = expert_data_minibatch['observations']
            batch_y = expert_data_minibatch['actions']

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            if training_iter_num % display_iter == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                print("Iter " + str(training_iter_num) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            training_iter_num += 1

        print("Optimization Finished!")

        # Calculate accuracy
        test_expert_data = expert_data_gen_fn()
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: test_expert_data['observations'],
                                          y: test_expert_data['actions'],
                                          keep_prob: 1.}))

if __name__ == '__main__':
    main()
