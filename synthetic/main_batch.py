# Toy Experiment for Particle FTRL-based SGDM for Online Bayesian Inference

import os
import numpy as np
# import torch as tc
import tensorflow as tf
import matplotlib.pyplot as plt
from dynamics_2 import Dynamics
import time as tm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from get_logp import get_logp
import argparse
import math
from  tqdm import tqdm

parser = argparse.ArgumentParser()

# 150 rounds of sub-linear is same as 50 linear

parser.add_argument("--T", type=int, default=500)
parser.add_argument("--lambda_c", type=float, default=0.1)
parser.add_argument("--dnType", type=str, default='SVGD_t2_logp_i_diffusion') # choose from 'SVGD_t2_logp_i_diffusion', 'SVGD', 'LD'
parser.add_argument("--batchType", type=str, default='batch')
parser.add_argument("--stepType", type=str, default='1t') #
parser.add_argument("--batchsize", type=str, default='static') # choose from 'sub' or 'static'

# define the parameter of the
parser.add_argument("--c1", type=float, default=0.1)
parser.add_argument("--c2", type=float, default=0)
parser.add_argument("--c3", type=int, default=6)



args = parser.parse_args()

dnType = args.dnType
batchType = args.batchType
stepType = args.stepType
lambda_c = args.lambda_c
T_total = args.T
batchsize = args.batchsize
c1 = args.c1
c2 = args.c2
c3 = args.c3


# define the function use in experiments
# def pot1(z):
#     z = tc.transpose(z,0,1)
#     return .5*((tc.norm(z, p=2, dim=0)-3.)/.5)**2 - tc.log(tc.exp(-.5*((z[0]-3.)/.5)**2) + tc.exp(-.5*((z[0]+3.)/.5)**2))

def get_logp_tf(theta, X, num_particles, T):
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
    # X_all = X.reshape((len(X), 1))
    # X_all = tf.convert_to_tensor(X_all)
    X_all = X
    ll_constant = (1.0 / (4 * np.sqrt(np.pi)))
    for i in range(num_particles):
        # L = ll_constant * (tf.exp(-0.25 * tf.square(X_all - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all - (theta[i, 0] + theta[i, 1]))))
        L = ll_constant * (tf.exp(-0.25 * tf.square(X_all - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all - (theta[i, 0] + theta[i, 1]))))
        temp1 = tf.reduce_sum(tf.log(L), keepdims=True)
        if i == 0:
            log_likelihood = temp1
        else:
            log_likelihood = tf.concat([log_likelihood, temp1], axis=1)
    return prior + log_likelihood

def get_log_prior_tf(theta):
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
    return prior

def get_log_likelihood_tf(theta, X, num_particles, t):
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

    # X_all = X.reshape((len(X), 1))
    # X_all = tf.convert_to_tensor(X_all)
    X_all = X
    ll_constant = (1.0 / (4 * np.sqrt(np.pi)))
    for i in range(num_particles):
        # L = ll_constant * (tf.exp(-0.25 * tf.square(X_all[-1] - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all[-1] - (theta[i, 0] + theta[i, 1]))))
        L = ll_constant * (tf.exp(-0.25 * tf.square(X_all - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all - (theta[i, 0] + theta[i, 1]))))
        # we can broadcast theta as theta only represent for the possibility
        temp1 = tf.reshape(tf.log(L), (1, 1))
        # temp1 = tf.reduce_sum(tf.log(L), keepdims=True)
        if i == 0:
            log_likelihood = temp1
        else:
            log_likelihood = tf.concat([log_likelihood, temp1], axis=1)

    return log_likelihood

def get_log_likelihood_batch(theta, X, num_particles):
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

    # X_all = X.reshape((len(X), 1))
    # X_all = tf.convert_to_tensor(X_all)
    X_all = X
    ll_constant = (1.0 / (4 * np.sqrt(np.pi)))
    for i in range(num_particles):
        # L = ll_constant * (tf.exp(-0.25 * tf.square(X_all[-1] - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all[-1] - (theta[i, 0] + theta[i, 1]))))
        L = ll_constant * (tf.exp(-0.25 * tf.square(X_all - theta[i, 0])) + tf.exp(-0.25 * tf.square(X_all - (theta[i, 0] + theta[i, 1]))))
        # we can broadcast theta as theta only represent for the possibility
        # temp1 = tf.reshape(tf.log(L), (1, 1))
        temp1 = tf.reduce_sum(tf.log(L), keepdims=True)
        if i == 0:
            log_likelihood = temp1
        else:
            log_likelihood = tf.concat([log_likelihood, temp1], axis=1)

    return log_likelihood

np.random.seed(23)
# experiment parameters
num_particles = 100
num_latent = 2
num_iter = 50
num_fig = 2
savefig = False
plt_range_limit_x = [-1.5, 2.5]
plt_range_limit_y = [-2, 2]
plt_num_points = 50

theta1 = 0
theta2 = 1
sigmax_sq = 2
T = args.T
epochs = 15
batch_sum = 0
samples0 = np.random.randn(num_particles, num_latent)
samples_tf = tf.Variable(samples0)

# samples_tc = tc.Variable(samples0)
# generate data points x_i follow 5.1 SGLD paper

X = np.zeros(20000) # data points shape = (T, 1)
for i in range(20000):
        u = np.random.random()
        if u < 0.5:
            X[i] = np.random.normal(theta1, np.sqrt(sigmax_sq))
        else:
            X[i] = np.random.normal(theta1 + theta2, np.sqrt(sigmax_sq))

# X = np.array([0.18668,-1.18406,1.14494,-0.23150,1.25413,-0.30881,-0.93575,-1.91542,1.26174,3.36283,-0.61791,0.09456,0.16105,-1.97060,-0.21477,1.12594,1.36874,0.28691,2.09972,3.01191,-2.48016,2.08469,0.85895,0.43451,-2.12251,-0.60040,1.63561,0.18895,-0.00979,2.64903,-2.31252,-0.31999,0.45862,0.04241,0.12309,-0.68630,-0.94139,-0.22246,1.29593,0.79667,-1.46483,2.26756,-1.15188,-0.55351,-1.96612,-0.21026,0.40006,0.93664,0.88029,0.59029,1.22570,0.53928,-0.07547,1.86272,1.16400,2.70949,0.40485,1.84971,0.46373,-3.11424,-1.10266,0.75165,-1.97135,-0.75634,2.13389,-0.41824,-2.10595,4.26532,1.52709,0.35159,-1.08596,0.88606,-1.05565,0.21335,-1.43228,-2.03355,1.57186,0.30954,2.79512,-0.73197,0.62561,-0.13210,2.15531,-1.63577,-0.06108,-0.38120,3.29082,2.76091,1.03750,0.28588,-1.68894,3.69985,-0.34551,1.08082,1.07356,-0.87367,-1.12656,2.14837,4.12616,-0.48807,-0.87276,1.33636,-4.46113,2.17897,2.24312,2.09508,2.11768,2.18546,-0.87284,0.90016,-0.91690,-0.71219,0.55724,0.34079,0.14977,1.08352,1.45052,0.47627,2.78601,1.05547,0.34260,2.79118,3.56668,1.78559,2.03555,1.58986,3.55883,-1.90465,1.27024,-0.72055,1.24844,2.72734,1.09118,-1.03997,1.95610,-0.14744,1.84559,-0.73512,3.66138,1.00525,1.82782,2.21462,2.23927,0.16729,-1.41698,-0.11586,1.26881,4.37820,-0.26527,1.51658,0.67227,0.33914,-0.77643,1.85255,1.07408,0.79435,1.58753,1.43757,-1.03154,-0.59101,2.60030,1.62561,-1.39286,-0.39880,1.70128,0.55636,-0.97453,0.96485,2.67787,0.12665,1.99446,1.30264,1.84387,-0.65339,0.35296,1.77258,0.22672,-0.75637,1.00727,1.63170,3.87424,-0.60599,2.93963,3.01077,-0.23555,0.58393,1.61282,1.09999,0.35548,3.70233,-2.63790,-0.24535,4.76431,-1.36591,0.17715,-2.04417,-2.84633,-0.09748,2.19414,-3.06425,-1.17115,-0.50489,0.97895,0.14442,0.37911,-2.07736,2.38679,1.96326,0.54128,-0.61567,2.07866,1.65199,2.53864,1.19405,-2.86922,3.11716,-0.59015,-0.20765,1.11745,0.46031,0.80663,0.37220,-1.61624,0.86581,3.17090,1.45829,4.16783,1.84463,3.85481,0.25511,2.14900,2.96705,-0.14532,1.26760,4.28512,1.94699,0.78151,-0.05011,-0.82936,0.90061,2.92595,1.78005,-0.78352,-1.89438,0.45114,-0.10808,-0.18353,1.38607,-0.02455,0.06549,-0.57322,0.75193,-0.49350,1.50085,0.59730,-1.02228,2.06947,1.67096,1.09503,0.60330,0.04187,-2.55459,-1.57126,-0.41698,-1.35427,0.17193,-0.99750,-0.27181,-0.07803,1.20623,-3.74936,-2.67412,-1.23053,3.68712,-1.31343,2.73514,-0.48534,-0.76012,1.75565,0.56027,0.30974,2.86331,-2.50055,-0.49032,0.77741,3.60195,0.29225,3.94267,2.39335,-3.34364,1.27851,2.15337,0.37216,-0.09375,1.93513,-0.73062,-2.44429,-3.71090,0.14417,0.70344,0.48058,-0.34734,4.31624,3.09148,2.34764,0.07678,2.04143,-0.60425,-0.77636,2.61704,-0.64702,2.64189,0.26752,-0.95801,0.00694,0.71869,4.44543,-0.86679,1.06672,-1.58906,1.28119,-0.40638,1.84478,-2.44173,1.10948,2.14663,0.14547,1.05891,2.63077,1.48207,0.51049,1.75899,1.09840,-0.06281,2.14265,2.16642,2.03655,1.51795,0.49190,0.67219,0.02671,2.16740,-1.30897,0.15269,0.78058,1.97869,0.66865,0.50958,-1.89820,0.68166,-2.94822,2.36255,0.46462,3.78490,3.42567,1.43355,2.93429,1.32255,-0.46908,-0.61828,2.82308,0.94519,-0.23747,2.58132,2.83075,-1.12132,-0.37805,1.03295,-0.70697,-0.23605,0.86790,1.02815,1.00592,-0.58354,2.59228,-1.51163,0.87935,-0.90491,-1.37038,-1.25390,-0.41393,0.68526,3.81462,-2.61049,-0.98640,0.69218,1.14332,1.27070,2.72579,0.79938,2.57685,0.91241,0.81448,0.52448,0.83175,2.33029,1.21030,-0.01582,-0.08355,2.47383,-1.12709,1.79472,1.28757,1.17215,-1.71907,1.21220,-2.96952,0.11671,1.66579,0.06425,-0.53028,2.27645,2.68987,-0.06420,-0.10496,0.15232,-0.45184,-1.18000,0.97046,2.67806,1.10553,-0.68135,1.29763,1.44404,2.68623,0.30603,0.33776,2.82747,-3.65655,1.04376,1.58590,-0.02246,-0.94911,1.96800,0.58560,0.42966,0.54845,-1.05179,3.22706,-2.44778,-2.03625,1.06282,0.21594,1.02207,2.02060,1.25173,1.51646,-0.76554,-1.11384,1.23959,2.38116,2.50542,-0.13699,-0.48667,2.58966,-1.40124,0.80982,0.62353,0.33041,0.64208,-0.75442,4.04240,-2.23725,1.80381,1.66672,-0.29990,2.08024,-0.27425,-1.41826,0.28654,1.00242,0.51459,-2.91841,2.38010,1.19653,0.69762,-0.89826,-0.01586,1.72464,1.63717,3.41669,0.57768,0.95278,0.10880,0.01498,-0.42618,1.03807,-0.45896,0.08433,0.87876,-2.68018,-1.78459,0.88549,0.99482,1.38343,1.28485,1.51496,-0.84644,0.52371,1.62808,2.17551,0.77423,1.38382,0.41823,-0.80320,0.64472,-0.27044,0.36894,1.06262,-0.61345,0.60468,-0.42926,-1.95347,-0.97023,1.94583,1.07397,0.66278,-0.68143,-1.55496,0.58716,-0.74347,0.29345,1.04450,1.04889,2.45792,0.38469,3.09462,0.08657,1.93199,1.28599,-1.53953,-0.60806,1.35251,0.28339,2.39498,-0.13213,-0.44781,0.66282,1.18421,1.98212,0.03534,-0.17229,1.38958,0.28197,1.90998,-0.73666,-2.01931,-0.52218,-0.34875,1.49885,1.40381,0.32145,-0.05737,-2.58481,-0.38106,2.35526,-1.83872,0.19914,1.17253,4.87287,1.56164,1.84447,0.94314,-0.41483,2.53283,1.29512,-0.15497,1.17775,-0.27729,0.85577,3.17658,-0.69263,0.59844,0.35256,0.52742,-0.43410,-0.49937,2.36883,1.80210,3.28331,2.91805,2.25942,2.99119,3.29126,2.15968,1.49649,1.69314,-0.86321,0.20138,1.04893,0.77883,-0.32596,0.96231,1.19969,-0.28969,-1.43060,-1.94166,0.61125,-0.82282,1.54825,-1.08131,-0.58453,0.04506,1.71688,4.76793,1.38883,1.32749,-0.25520,1.36524,-1.18122,-0.67790,-0.08735,1.34060,-1.92263,-0.06747,0.28679,-0.61701,0.59324,1.29894,-0.59691,1.82190,-0.16941,3.26623,1.50423,1.82399,1.00793,-1.01412,3.53825,2.46717,0.17450,-1.73431,-0.47567,-0.19978,2.73830,-1.74636,-2.71801,0.81304,3.40708,1.05563,1.03131,-1.12025,2.39818,3.97271,0.36033,1.33652,1.75776,-2.37026,-0.47418,0.49872,0.57743,0.63965,1.38047,-2.98236,-0.53623,0.83033,0.03434,-0.29094,1.28110,2.55195,0.01539,0.36708,2.26686,0.49661,1.33120,1.84358,-0.67349,0.43524,0.11363,-0.58605,0.67259,0.59963,0.44832,0.74016,1.04049,-0.55501,2.41115,2.70725,-0.07774,2.83827,2.08173,2.20645,-1.28268,1.59475,-1.49880,-0.15425,1.34131,-0.07861,2.44805,3.61881,-0.41142,2.45965,2.34810,-3.81630,2.82755,-0.69326,1.54459,-0.72258,-0.48620,0.61173,-0.08819,0.59822,4.21320,-1.64121,2.82442,-0.18152,0.25707,0.28531,0.38728,1.14931,0.02749,0.33209,-1.08609,-0.96625,-1.71943,-0.94500,0.00239,-2.87173,1.67161,2.87131,0.66214,-0.29105,-0.17689,0.25628,0.69437,1.60469,1.72210,-2.50540,0.48479,0.56835,-1.30235,-0.85583,-2.56196,2.63627,-0.58224,2.14668,0.48039,-1.07130,-0.10339,1.34068,0.29887,-0.83700,0.49408,0.65798,-1.06042,3.34848,0.53705,-2.03742,-0.00071,-0.72377,0.63764,2.17030,0.47318,0.08177,1.91084,3.72833,-0.42054,2.18337,0.82163,3.18305,-0.65745,0.05543,4.08846,-0.43395,-0.50057,0.98604,-0.58375,-1.44608,0.59428,1.53400,1.28876,-0.03358,-0.41868,1.71478,1.48179,-1.26925,1.90739,-0.09190,2.26760,0.66609,-2.16504,-1.55053,-0.81903,-1.78539,0.92303,3.75493,2.51277,1.71032,1.15226,-0.75711,2.39577,-0.30997,-0.10672,-1.13927,1.71715,-0.37395,0.72650,-0.38010,-2.16938,1.36624,0.68227,-1.35403,-1.19837,-0.54827,2.43848,0.29659,0.93042,0.90361,-0.25098,2.05473,-0.10411,1.48528,-1.17691,0.87478,2.27937,3.28528,1.72979,-0.78217,0.72971,-0.75190,2.06029,0.99700,2.30005,2.39101,0.66153,1.07725,-1.60548,0.80242,0.80211,1.27096,2.67499,3.69361,-0.78296,0.43792,1.47377,1.52660,2.15718,-0.66540,-0.90749,1.34550,0.46169,0.25942,1.69954,1.60432,-2.57218,-0.68901,0.73272,1.15838,0.39725,1.17925,0.54815,-0.08871,1.68884,1.84593,-0.76341,0.49381,-0.09889,1.04620,1.15978,2.72911,-0.95443,1.14171,1.79562,0.24977,0.37348,0.84463,1.33502,-0.10642,1.66145,-0.36177,-1.13192,-0.45227,1.20094,1.85882,-1.17665,0.51655,0.73008,0.00618,0.39463,1.14205,3.18872,-0.72617,1.97288,0.53986,-1.99611,0.50742,1.67489,3.01097,1.79588,2.09277,1.28297,1.34748,-0.76142,0.67140,1.40222,1.89003,1.01402,0.69321,-0.86324,-1.29561,0.42752,-1.78838,1.61980,2.06949,3.59112,-0.25799,-2.58413,1.61298,-1.22487,0.68313,-0.19161,0.68950,1.69207,1.47030,-1.15470,1.66288,0.02610,0.26427,1.96375,3.81801,1.99276,0.41806,-0.41886,0.64641,0.04850,-1.24998,-0.67001,-0.54303,0.55278,0.42896,-1.35788,1.24017,0.35948,-0.16252,0.48709,1.09916,0.06193,1.27966,3.55361,-1.75221,0.21229,1.70103,0.33703,0.33375,1.52833,2.73941,1.90312,0.46805,-0.01616,0.87631,-1.24758,0.91719,0.15040,3.93409,2.84830,2.39765,-0.60707,-2.31682,-0.29307,2.78429,0.15207,-1.31009,0.81730,-0.82199,1.34913,0.31433,1.50631,2.50623,0.68850,2.26103,-1.19300,0.49512,0.02728,0.89783,0.59968,0.43925,2.23387,-0.58840,0.42156,1.15925,1.84054,3.05566,1.56540,2.72212,1.85984,3.00915,0.58018,-0.68987,0.62763,1.40389,1.02497,2.52220,-0.06249,-0.13391,-1.08791,-3.40837,-0.64235,-0.52359,-0.26526,1.38518,1.88475])


# draw

# grid_side = np.linspace(plt_range_limit[0], plt_range_limit[1], plt_num_points)
# mesh_z1, mesh_z2 = np.meshgrid(grid_side, grid_side)
# z_a = np.zeros((plt_num_points, plt_num_points))
# for i in range(plt_num_points):
#     for j in range(plt_num_points):
#         theta = np.array([[mesh_z1[i, j]], [mesh_z2[i, j]]])
#         z_a[i,j] = get_logp(theta, X[:t], T)
# z_a = np.exp(z_a)

# print('zv:', zv)
# prob_tc = get_logp(zv, X[:t], T)
# print('prob_tc:', prob_tc)
# prob_tc = tc.reshape(prob_tc, (plt_num_points, plt_num_points))
# print('prob_tc reshape:', prob_tc)
# z_min, z_max = -np.abs(prob_tc).max(), np.abs(prob_tc).max()



# new method
# grid_side_1 = np.linspace(plt_range_limit[0], plt_range_limit[1], plt_num_points)
# mesh_z1_1, mesh_z2_1 = np.meshgrid(grid_side_1, grid_side_1)
# zv_1 = np.zeros((plt_num_points, plt_num_points))
# for i in range(plt_num_points):
#     for j in range(plt_num_points):
#         theta = np.array( [[mesh_z1_1[i,j]],[mesh_z2_1[i,j]]] )
#         zv_1[i,j] = log_f(theta, X)


# old method
grid_side_x = np.linspace(plt_range_limit_x[0], plt_range_limit_x[1], plt_num_points)
grid_side_y = np.linspace(plt_range_limit_y[0], plt_range_limit_y[1], plt_num_points)
mesh_z1, mesh_z2 = np.meshgrid(grid_side_x, grid_side_y)
zv = np.hstack([mesh_z1.reshape(-1, 1), mesh_z2.reshape(-1, 1)])



# zp_tf  = tf.placeholder(tf.float64, [None, num_latent])
# X_tf = tf.placeholder(tf.float64, [1, t])
# num_tf = tf.placeholder(tf.float64, [])
# T_tf = tf.placeholder(tf.float64)
# zp_tf = tf.cast(zp_tf, dtype=tf.float64)

total_data_num = 0
for i in range(T):
    if batchsize == 'sub':
        total_data_num += math.ceil(i ** 0.55)
    elif batchsize == 'lin':
        total_data_num += math.ceil(i)
    elif batchsize == 'static':
        total_data_num += 20

prob_tf = get_logp(zv, X[:total_data_num], num_particles=plt_num_points**2, T=T)
# prob_tf = get_logp(zp_tf)
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    phat_z = sess.run(prob_tf)
#     print('prob_tf:', phat_z)
phat_z = phat_z.reshape([plt_num_points, plt_num_points])
z_min, z_max = -np.abs(phat_z).max(), np.abs(phat_z).max()
phat_z_tmp = phat_z
def showfig(pm):
    """
train the particles(samples here) and draw the pictures for particles and posterior
    :param pm: class pm:
    dnType = ; dnNormalize =
    accType =
    optType = ; stepsize =
    bwType =
    """

    X_online = tf.placeholder(tf.float64)
    X_tf = tf.placeholder(tf.float64)
    T_tf = tf.placeholder(tf.float64)
    op_samples, dninfo = Dynamics(pm.dnType, pm).evolve(samples_tf, X_tf, X_online, T_tf, num_particles = num_particles, get_logp=get_logp_tf, get_log_likelihood=get_log_likelihood_tf, get_log_prior=get_log_prior_tf, get_log_likelihood_batch=get_log_likelihood_batch)
    fig, ax = plt.subplots(1, num_fig, sharey=True, sharex=True, figsize=(6, 6))
    # fig, ax = plt.subplots(1, 1)
    if num_fig == 1:
        ax = [ax]
    else:
        figrange = range(1, num_fig)
        ax[0].set_xlim(plt_range_limit_x);
        ax[0].set_ylim(plt_range_limit_y);
        ax[0].set_aspect('equal');
        # ax[0].set_xticks([]);
        # ax[0].set_yticks([])
        # ax[0].contour(mesh_z1, mesh_z2, z_a, cmap='Blues')
        ax[0].contour(mesh_z1, mesh_z2, phat_z, 300, zorder=10)
        ax[0].scatter(samples0[:, 0], samples0[:, 1], c="dimgrey", marker="o", s=20, linewidths=0, edgecolor='white',alpha=1, zorder=20)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # tf.random.shuffle(X)
        batch_sum = 0
        for i in figrange:
            # for batch in range(batch_num):
            #     for tau in range(batch_size):
            #         step_size = 1 / batch_size
            #         samples = sess.run([op_samples, dninfo.L_samples], feed_dict={X_tf: X[:T].reshape(T, 1), X_online: X[:delta_t].reshape(delta_t, 1), T_tf: t})[1][0]
            #
            # batch learning with fixed batch-size and restarting step-size
            # for batch in range(batch_num):
            #     for t in range(batch_size):
            #         tf.random.shuffle(X)
            #         samples = sess.run([op_samples, dninfo.L_samples], feed_dict={X_tf: X[:T].reshape(T, 1), X_online: X[batch * batch_size + t].reshape(1, 1), T_tf: t})[1][0]
            #         print(f'batch {batch} time {t} mean: {sess.run(tf.reduce_mean(samples, 0))}, variance: {sess.run(tf.math.reduce_variance(samples, 0))}')

            for t in range(1, T-1):
                if batchsize == 'sub':
                    batch_size = math.ceil(t ** 0.55)
                elif batchsize == 'lin':
                    batch_size = math.ceil(t)
                elif batchsize == 'static':
                    batch_size = 20

                idx_begin = batch_sum
                idx_end = batch_sum + batch_size
                batch_sum = idx_end
                sess.run(op_samples, feed_dict={X_tf: X[:T].reshape(T, 1),
                                                X_online: X[idx_begin:idx_end].reshape(idx_end - idx_begin, 1),
                                                T_tf: t})
                print(f'time {t} idx_begin {idx_begin} idx_end {idx_end}')
                # for epoch in range(1, batch_size + 1):
                # # for epoch in range(1, 10):
                #     if batchType == 'full':
                #         sess.run(op_samples,
                #                  feed_dict={X_tf: X.reshape(5000, 1), X_online: X[t].reshape(1, 1), T_tf: t})
                #     elif batchType == 'batch':
                #         # sess.run(op_samples, feed_dict={X_tf: X[:T].reshape(T, 1), X_online: X[t].reshape(1, 1), T_tf: t})
                #         sess.run(op_samples, feed_dict={X_tf: X[:T].reshape(T, 1),X_online: X[idx_begin:idx_end].reshape(idx_end - idx_begin, 1), T_tf: epoch})
                #         # samples = sess.run([op_samples, dninfo.L_samples],
                #         #                    feed_dict={X_tf: X[:T].reshape(T, 1), X_online: X[t].reshape(1, 1),
                #         #                               T_tf: t})[1][0]
                #     elif batchType == 'online':
                #         sess.run(op_samples, feed_dict={X_tf: X[:T].reshape(T, 1),X_online: X[idx_begin + epoch - 1:idx_begin + epoch].reshape(1, 1), T_tf: epoch})
                #     print(f'time {t} epoch {epoch}')
                #     # print(dnType)
                #     # print(f'time {t} mean: {sess.run(tf.reduce_mean(samples, 0))}, variance: {sess.run(tf.math.reduce_variance(samples, 0))}')
                #     # print(f'time {t} true_mean: {sess.run(tf.reduce_mean(X[:delta_t].reshape(delta_t, 1), 0))}, true_variance: {sess.run(tf.math.reduce_variance(X[:delta_t].reshape(delta_t, 1), 0))}')
                # #


            samples = sess.run([op_samples, dninfo.L_samples], feed_dict={X_tf: X[:(T-1)].reshape(T-1,1), X_online:X[T-1].reshape(1,1), T_tf: T})[1][0]
            ax[i].set_aspect('equal'); ax[i].set_xticks([]); ax[i].set_yticks([])
            #ax[i].pcolormesh(mesh_z1, mesh_z2, phat_z, cmap='RdBu', vmin=z_min, vmax=z_max)
            ax[i].contour(mesh_z1, mesh_z2, phat_z_tmp, 300, zorder=10)
            ax[i].scatter(samples[:,0], samples[:,1], c='dimgrey', marker="o", s=30, linewidths=0, edgecolor='white', alpha=1, zorder=20)
            print(dnType)
            print(f'test case {dnType} {T} {stepType} {lambda_c} {batchType} mean: {sess.run(tf.reduce_mean(samples, 0))}, variance: {sess.run(tf.math.reduce_variance(samples, 0))}')
    return fig, ax

class PM:
    dnType = args.dnType
    accType = 'wgd'
    optType = 'gd'
    stepsize = 5e-1
    bwType = 'med'
    dnNormalize = True
    stepsize_type = stepType
    lambda_c = lambda_c
    batchType = batchType
    c1 = c1
    c2 = c2
    c3 = c3
#    stepsize_type:  'con' or 'apt'


t0 = tm.time()
fig, ax = showfig(PM())
# fig.show()
# fig.savefig(str(dnType) + '_' + str(T) + '_' + str(stepType) + '_' + str(batchType) + '.png', format='png')
fig.savefig('batch1' + str(dnType) + '_' + str(T) + '_' + str(T) + '_' + str(stepType) + '_' + str(lambda_c) + '_' + str(c1) + '_' + str(c2) + '_' + str(c3) + '.png', format='png')
t1 = tm.time()
print(t1 - t0)
