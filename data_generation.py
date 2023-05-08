import pdb

from jax.config import config
config.update("jax_enable_x64", True)


from jax import jit, vmap, value_and_grad
from jax.lax import scan
import jax.numpy as jnp
from jax.numpy.linalg import inv
import jax.random as jrandom
import matplotlib.pyplot as plt

from func_estimators import init_nica_params, nica_mlp
from utils import gaussian_sample_from_mu_prec, invmp, tree_prepend
from utils import multi_tree_stack


import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST

# import tensorflow_datasets as tfds
from flax import linen as nn
# from flax.metrics import tensorboard
# from flax.training import train_state
import optax



def ar2_lds(alpha, beta, mu, vz, vz0, vx):
    B = jnp.array([[1 + alpha - beta, -alpha], [1.0, .0]])
    b = jnp.array([beta * mu, .0])
    b0 = jnp.array([.0, .0])
    C = jnp.array([[1., .0]])
    c = jnp.array([0.])
    Q = jnp.diag(1 / jnp.array([vz, 1e-3 ** 2]))
    Q0 = jnp.diag(1 / jnp.array([vz0, vz0]))
    R = jnp.diag(1 / jnp.array([vx]))
    return B, b, b0, C, c, Q, Q0, R


def generate_oscillatory_lds(key):
    keys = jrandom.split(key, 3)
    alpha = jrandom.uniform(keys[0], minval=.6, maxval=.8)  # momentum
    beta = .1  # mean reversion
    mu = jrandom.uniform(keys[1], minval=-1., maxval=1.)  # mean level
    vz = jrandom.uniform(keys[2], minval=.1**2, maxval=.2**2)  # driving noise variance
    vz0 = .1  # initial state variance
    vx = .1**2  # observation noise variance
    return ar2_lds(alpha, beta, mu, vz, vz0, vx)


def generate_meanreverting_lds(key):
    alpha = .1  # momentum
    keys = jrandom.split(key, 2)
    beta = jrandom.uniform(keys[0], minval=.7, maxval=.9)  # mean reversion
    mu = jrandom.uniform(keys[1], minval=-3., maxval=3.)  # mean level
    vz = .2**2  # driving noise variance
    vz0 = .1  # initial state variance
    vx = .1**2  # observation noise variance
    return ar2_lds(alpha, beta, mu, vz, vz0, vx)


def generate_slds(K, key):
    os_key, mr_key = jrandom.split(key)
    A = jnp.array([[.99, .01], [.01, .99]])
    a0 = jnp.ones(K)/K
    B, b, b0, C, c, Q, Q0, R = multi_tree_stack([
        generate_oscillatory_lds(os_key),
        generate_meanreverting_lds(mr_key)])
    return (A, a0), (B, b, b0, C, c, Q, Q0, R)


def gen_markov_chain(pi, A, num_steps, samplekey):
    def make_mc_step(A, num_states):
        @jit
        def sample_mc_step(cur_state, key):
            p = A[cur_state]
            new_state = jrandom.choice(key, jnp.arange(num_states), p=p)
            return new_state, new_state
        return sample_mc_step

    K = pi.shape[0]
    keys = jrandom.split(samplekey, num_steps)
    start_state = jrandom.choice(keys[0], jnp.arange(K), p=pi)
    mc_step = make_mc_step(A, K)
    _, states = scan(mc_step, start_state, keys[1:])
    states = jnp.concatenate([start_state.reshape((1,)), states])
    return states


def make_slds_sampler(B, b, Q):
    @jit
    def sample_slds(z_prev, state_with_key):
        state, key = state_with_key
        z_mu = B[state]@z_prev + b[state]
        z = gaussian_sample_from_mu_prec(z_mu, Q[state], key)
        return z, (z, z_mu)
    return sample_slds


def gen_slds(T, K, d, paramkey, samplekey):
    paramkey, p_key = jrandom.split(paramkey)
    s_ldskey, s_hmmkey = jrandom.split(samplekey)

    # generate variables (return mean and variances of generation)
    (A, pi), (B, b, b_init, _, _, Q, Q_init, _) = generate_slds(K, p_key)

    # generate hidden markov chain (return all discrete states)
    states = gen_markov_chain(pi, A, T, s_hmmkey)

    # sample slds (use reparam trick with mean and precision)
    s_ldskeys = jrandom.split(s_ldskey, T)
    z_init = gaussian_sample_from_mu_prec(b_init[states[0]],
                                          Q_init[states[0]], s_ldskeys[0])
    # make a function to sample using reparam trick and then scan to sample whole chain
    sample_func = make_slds_sampler(B, b, Q)
    z, z_mu = tree_prepend((z_init, b_init[states[0]]),
                           scan(sample_func, z_init,
                                (states[1:], s_ldskeys[1:]))[1])
    hmm_params = (pi, A)
    lds_params = (b_init, Q_init, B, b, Q)
    return z, z_mu, states, lds_params, hmm_params


def gen_slds_nica(N, M, T, K, d, L, paramkey, samplekey, repeat_layers=False):
    # generate several slds
    '''
    N: number of ICs
    M: dimension of observed data
    T: number of timesteps
    K: number of HMM states. Fixed at 2 in experients
    d: dimension of lds state. Fixed at 2 in experim
    L: number of nonlinear layers; 0 = linear ICA
    '''

    # Define our dataset, using torch datasets
    batch_size = 128
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

    # Get the full train dataset (for checking accuracy while training)
    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    # train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)
    train_labels = np.array(mnist_dataset.train_labels)

    # Get full test dataset
    mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
    test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1),
                            dtype=jnp.float32)
    # test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

    
    paramkeys = jrandom.split(paramkey, N+1)
    samplekeys = jrandom.split(samplekey, N+1)
    z, z_mu, states, lds_params, hmm_params = vmap(
        gen_slds, (None, None, None, 0, 0))(T, K, d, paramkeys[1:],
                                            samplekeys[1:])
    s = z[:, :, 0]
    # mix signals
    paramkeys = jrandom.split(paramkeys[0], 2)
    nica_params = init_nica_params(N, M, L, paramkeys[0], repeat_layers)
    f = vmap(nica_mlp, (None, 1), 1)(nica_params, s)
    # add appropriately scaled output noise (R is precision!)
    R = inv(jnp.eye(M)*jnp.diag(jnp.cov(f))*0.15)
    likelihood_params = (nica_params, R)
    x_keys = jrandom.split(samplekeys[0], T)
    x = vmap(gaussian_sample_from_mu_prec, (1, None, 0), 1)(f, R, x_keys)
    # double-check variance levels on output noise
    Rxvar_ratio = jnp.mean(jnp.diag(invmp(R, jnp.eye(R.shape[0]))
                                    / jnp.cov(x)))
    print(' *inv(R)/xvar: ', Rxvar_ratio)
    print('obs shape',x.shape)
    return x, f, z, z_mu, states, likelihood_params, lds_params, hmm_params


def gen_MNIST(N, M, T, K, d, L, paramkey, samplekey, repeat_layers=False):
    # generate several slds
    '''
    N: number of ICs
    M: dimension of observed data
    T: number of timesteps
    K: number of HMM states. Fixed at 2 in experients
    d: dimension of lds state. Fixed at 2 in experim
    L: number of nonlinear layers; 0 = linear ICA
    '''

    # Define our dataset, using torch datasets
    batch_size = 128
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

    # Get the full train dataset (for checking accuracy while training)
    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    # train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)
    train_labels = np.array(mnist_dataset.train_labels)

    # Get full test dataset
    mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
    test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1),
                            dtype=jnp.float32)
    # test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)


if __name__ == "__main__":
    for i in range(10):
        key = jrandom.PRNGKey(i)
        N = 3
        M = 12
        T = 100000
        d = 2
        K = 2
        L = 1
        #stay_prob = 0.95

        pkey, skey = jrandom.split(key)

        # generate data from slds with linear ica
        if L == 0:
            x, z, z_mu, states, x_params, z_params, u_params = gen_slds_linear_ica(
                N, M, T, K, d, pkey, skey)
        elif L > 0:
        # with nonlinear ICA
            x, z, z_mu, states, x_params, z_params, u_params = gen_slds_nica(
                N, M, T, K, d, L, pkey, skey)

    ## sanity check to make sure Q reasonable level
    b0, Q0, B, b, Q = z_params
    for i in range(N):
        for j in range(K):
            print("* QZvar_ratio:",
                  jnp.diag(inv(Q)[i, j]
                           / jnp.cov(z[i][states[i] == j].T)).mean())

    for i in range(N):
        print("z lagcorr:", jnp.corrcoef(z[i, :-10, :].T, z[i, 10:, :].T))
    print("x lagcorr:", round(jnp.corrcoef(x[:, :-1], x[:, 1:]), 2))

    plt.plot(z_mu[:, :200, 0].T)
    plt.show()
    pdb.set_trace()






############################################################################################################
# ADDED FOR MNIST
############################################################################################################

# def get_datasets():
#   """Load MNIST train and test datasets into memory."""
#   ds_builder = tfds.builder('mnist')
#   ds_builder.download_and_prepare()
#   train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
#   test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
#   train_ds['image'] = jnp.float32(train_ds['image']) / 255.
#   test_ds['image'] = jnp.float32(test_ds['image']) / 255.
#   return train_ds, test_ds


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


@jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images)
    one_hot = nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)




def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))