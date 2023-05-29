import argparse
import pdb
import sys

from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as jrandom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data_generation import gen_slds_nica, make_cifar10_maze_trajectory, make_MNIST_maze_trajectory
from train import full_train, full_train_MNIST

import torch
import os, json, pickle
# uncomment to debug NaNs
#config.update("jax_debug_nans", True)

sys.path.append("./../")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-s', type=int, default=10,
                        help="subfolder to save in")
    parser.add_argument('-batch_size', type=int, default=1000,
                        help="size of batch")
    parser.add_argument('-n', type=int, default=1, # REPLACED 3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                        help="number of ICs")
    # parser.add_argument('-m', type=int, default=12,
    #                     help="dimension of observed data")
    parser.add_argument('-t', type=int, default=1024,
                        help="number of timesteps")
    # parser.add_argument('-l', type=int, default=1,
    #                     help="number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('-d', type=int, default=10, # REPLACE 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                        help="dimension of lds state. Fixed at 2 in experim.")
    parser.add_argument('-k', type=int, default=10, # REPLACE 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                        help="number of HMM states. Fixed at 2 in experients")
    parser.add_argument('--whiten', action='store_true', default=False,
                        help="PCA whiten data as preprocessing")
    parser.add_argument('--gt-gm-params', action='store_true', default=False,
                        help="debug with GM parameters at ground truth")
    # set seeds
    parser.add_argument('--param-seed', type=int, default=50,
                        help="seed for initializing data generation params")
    parser.add_argument('--data-seed', type=int, default=1,
                        help="seed for initializing data generation sampling")
    parser.add_argument('--est-seed', type=int, default=99,
                        help="seed for initializing function estimators")
    # inference & training & optimization parameters
    parser.add_argument('--inference-iters', type=int, default=5,
                        help="num. of inference iterations")
    parser.add_argument('--num-samples', type=int, default=1,
                        help="num. of samples for elbo")
    # parser.add_argument('--hidden-units-enc', type=int, default=128,
    #                     help="num. of hidden units in encoder estimator MLP")
    # parser.add_argument('--hidden-units-dec', type=int, default=64,
    #                     help="num. of hidden units in decoder estimator MLP")
    # parser.add_argument('--hidden-layers-enc', type=int, default=2,
    #                     help="num. of hidden layers in encoder estimator MLP")
    # parser.add_argument('--hidden-layers-dec', type=int, default=1,
    #                     help="num. of hidden layers in decoder estimator MLP")
    parser.add_argument('--nn-learning-rate', type=float, default=1e-2,
                        help="learning rate for training function estimators")
    parser.add_argument('--gm-learning-rate', type=float, default=1e-2,
                        help="learning rate for training GM parameters")
    parser.add_argument('--burnin', type=float, default=500,
                        help="keep output precision fixed for _ iterations")
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument('--decay-rate', type=float, default=1.,
                        help="decay rate for training (default to no decay)")
    parser.add_argument('--decay-interval', type=int, default=1e10,
                        help="interval (in iterations) for full decay of LR")
    parser.add_argument('--plot-freq', type=int, default=10,
                        help="plotting frequency")
    # saving and loading
    # parser.add_argument('--out-dir', type=str, default="output/",
    #                     help="location where data is saved")
    parser.add_argument('--resume-best', action='store_true', default=False,
                        help="resume from best chkpoint for current args")
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help="eval only wihtout training")


    args = parser.parse_args()
    return args


def main():

    # print('jax config ', config.FLAGS.xla_python_client_preallocate)

    args = parse()

    ###########################################################################################################
    # Added

    ###############################################################################
    # get arguments
    ###############################################################################

    SLURM_ARRAY_TASK_ID = str(args.s) #sys.argv[1]
    print('SLURM_ARRAY_TASK_ID ', SLURM_ARRAY_TASK_ID)

    ARG_FILE_NAME = 'arguments_HMM_cifar_9_simple.json'
    parent_folder = '/nfs/gatsbystor/williamw/latent_confounder/'
    # parent_folder = '/home/william/mnt/gatsbystor/latent_confounder/'
    ARGUMENT_FILE = parent_folder + 'arg_files/' + ARG_FILE_NAME

    with open(ARGUMENT_FILE) as json_file:
        ARGS = json.load(json_file)
        print('PARAMETERS ', ARGS[SLURM_ARRAY_TASK_ID])
        paramDict = ARGS[SLURM_ARRAY_TASK_ID]

    OUTPUT_FOLDER = paramDict['MAIN_FOLDER'] + '/SNICA_' + paramDict['SUB_FOLDER']
    saveFolder = parent_folder + OUTPUT_FOLDER + '/'

    if paramDict['data_source'] == 'MNIST':
        data_folder = 'HMM_shared_data_MNIST_0/'
    elif paramDict['data_source'] == 'cifar10':
        data_folder = 'HMM_shared_data_cifar_1/'

    ###############################################################################
    # check if cuda is available
    ###############################################################################

    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda available')

    device = torch.device("cuda" if cuda else "cpu")

    # check if using CUDA
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if os.path.exists(saveFolder):
        print('overwrite')
    else:
        os.makedirs(saveFolder)

    # Data type: float64 / float32
    data_type = torch.float32
    torch.set_default_dtype(data_type)
    ###########################################################################################################



    #
    # # generate data
    # param_key = jrandom.PRNGKey(args.param_seed)
    # data_key = jrandom.PRNGKey(args.data_seed)
    #
    # # generate simulated data
    # # !BEWARE d=2, k=2 fixed in data generation code
    # x, f, z, z_mu, states, *params = gen_slds_nica(args.n, args.m, args.t,
    #                                                args.k, args.d, args.l,
    #                                                param_key, data_key,
    #                                                repeat_layers=True)
    # print('generated data',x.shape, z.shape, z_mu.shape, states.shape)

    ###########################################################################################################
    # Added
    timesteps = paramDict['timesteps']
    num_sequences = paramDict['num_sequences']
    if paramDict['saved_samples']:
        print('using saved samples')
        samples = pickle.load(open(parent_folder + data_folder + str(timesteps)+'_samples.pkl', 'rb'))
        true_states, true_transition_mat = samples['true_states'], samples['true_transition_mat']
    # obs_shuffle, obs_non_shuffle, true_states, true_transition_mat = make_MNIST_maze_trajectory(
    #     timesteps, true_states=true_states, true_transition_mat=true_transition_mat, batch_size=1,
    #     num_sequences=paramDict['num_sequences'])
    obs_shuffle, obs_non_shuffle, true_states, true_transition_mat = make_cifar10_maze_trajectory(
        timesteps, true_states=true_states, true_transition_mat=true_transition_mat, batch_size=1,
        num_sequences=paramDict['num_sequences'])

    args.t = timesteps
    args.num_sequences = num_sequences

    print('data is loaded')

    ###########################################################################################################

    # # we have not tried this option but could be useful in some cases
    # if args.whiten:
    #     pca = PCA(whiten=True)
    #     x = pca.fit_transform(x.T).T

    # # train
    # est_params, posteriors, best_elbo = full_train(x, f, z, z_mu, states,
    #                                                params, args, args.est_seed)

    args.saveFolder = saveFolder
    # train
    est_params, posteriors, best_elbo = full_train_MNIST(obs_shuffle, true_states, args, args.est_seed)


if __name__ == "__main__":
    sys.exit(main())
