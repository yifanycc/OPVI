import numpy as np
import tensorflow as tf


def get_logp(theta, X, num_particles, T):
    '''
    The function 'f' is the posterior:
        f(\theta) \propto p(\theta) * \prod_{i=1}^N p(x_i | \theta)
    :param num_particles:
    :param theta: tf.variable, shape = (num_particles, num_latent) (200, 2)
    :param X: data points at time slot t
    :return: log of posterior, shape = (1, 200)
    '''
    # scale = T / float(len(X))
    # theta = tf.transpose(theta)
    inverse_covariance = tf.constant([[0.1, 0], [0, 1]], dtype=tf.float64)
    prior_constant = 1.0 / (2 * np.pi * np.sqrt(10))
    temp = tf.matmul(theta, inverse_covariance)
    prior = tf.log(prior_constant) - 0.5 * tf.matmul(temp, tf.transpose(theta))
    prior = tf.diag_part(prior)

    X_all = X.reshape((len(X), 1))
    X_all = tf.convert_to_tensor(X_all)
    ll_constant = (1.0 / (4 * np.sqrt(np.pi)))
    for i in range(num_particles):
        L = ll_constant * (tf.exp(-0.25 * tf.square(X_all - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all - (theta[i, 0] + theta[i, 1]))))
        temp1 = tf.reduce_sum(tf.log(L), keepdims=True)
        if i == 0:
            log_likelihood = temp1
        else:
            log_likelihood = tf.concat([log_likelihood, temp1], axis=1)
    return prior + log_likelihood


def get_log_likelihood(theta, X, num_particles, T):
    '''
    The function 'f' is the posterior:
        f(\theta) \propto p(\theta) * \prod_{i=1}^N p(x_i | \theta)
    :param num_particles:
    :param theta: tf.variable, shape = (num_particles, num_latent) (200, 2)
    :param X: data points at time slot t
    :return: log of likelihood, shape = (1, 200)
    '''
    # scale = T / float(len(X))
    # theta = tf.transpose(theta)
    # inverse_covariance = tf.constant([[0.1, 0], [0, 1]], dtype=tf.float64)
    # prior_constant = 1.0 / (2 * np.pi * np.sqrt(10))
    # temp = tf.matmul(theta, inverse_covariance)
    # prior = tf.log(prior_constant) - 0.5 * tf.matmul(temp, tf.transpose(theta))
    # prior = tf.diag_part(prior)

    X_all = X.reshape((len(X), 1))
    X_all = tf.convert_to_tensor(X_all)
    ll_constant = (1.0 / (4 * np.sqrt(np.pi)))
    for i in range(num_particles):
        L = ll_constant * (tf.exp(-0.25 * tf.square(X_all - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all - (theta[i, 0] + theta[i, 1]))))
        temp1 = tf.reduce_sum(tf.log(L), keepdims=True)
        if i == 0:
            log_likelihood = temp1
        else:
            log_likelihood = tf.concat([log_likelihood, temp1], axis=1)
    return log_likelihood