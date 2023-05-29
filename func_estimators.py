from jax.config import config

config.update("jax_enable_x64", True)

import pdb

import jax.nn as nn
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, value_and_grad, jit
from jax.lax import scan
#from jax.ops import index, index_update
import optax

from flax import linen as fnn
from flax.training import train_state

def l2normalize(W, axis=0):
    """Normalizes MLP weight matrices.
    Args:
        W (matrix): weight matrix.
        axis (int): axis over which to normalize.
    Returns:
        Matrix l2 normalized over desired axis.
    """
    l2norm = jnp.sqrt(jnp.sum(W*W, axis, keepdims=True))
    W = W / l2norm
    return W


def smooth_leaky_relu(x, alpha=0.1):
    """Calculate smooth leaky ReLU on an input.
    Source: https://stats.stackexchange.com/questions/329776/ \
            approximating-leaky-relu-with-a-differentiable-function
    Args:
        x (float): input value.
        alpha (float): controls level of nonlinearity via slope.
    Returns:
        Value transformed by the smooth leaky ReLU.
    """
    return alpha*x + (1 - alpha)*jnp.logaddexp(x, 0)


def SmoothLeakyRelu(slope):
    """Smooth Leaky ReLU activation function.
    Args:
        slope (float): slope to control degree of non-linearity.
    Returns:
       Lambda function for computing smooth Leaky ReLU.
    """
    return lambda x: smooth_leaky_relu(x, alpha=slope)


def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x


def init_layer_params(in_dim, out_dim, key):
    W_key, b_key = jrandom.split(key, 2)
    W_init = nn.initializers.glorot_uniform(dtype=jnp.float64)
    b_init = nn.initializers.normal(dtype=jnp.float64)
    return W_init(W_key, (in_dim, out_dim)), b_init(b_key, (out_dim,))


def unif_nica_layer(N, M, key, iter_4_cond=1e4):
    def _gen_matrix(N, M, key):
        A = jrandom.uniform(key, (N, M), minval=0., maxval=2.) - 1.
        A = l2normalize(A)
        _cond = jnp.linalg.cond(A)
        return A, _cond

    # generate multiple matrices
    keys = jrandom.split(key, iter_4_cond)
    A, conds = vmap(_gen_matrix, (None, None, 0))(N, M, keys)
    target_cond = jnp.percentile(conds, 25)
    target_idx = jnp.argmin(jnp.abs(conds-target_cond))
    return A[target_idx]


def init_nica_params(N, M, nonlin_layers, key, repeat_layers):
    '''BEWARE: Assumes factorized distribution
        and equal width in all hidden layers'''
    print('func estimators REPLACED!')
    layer_sizes = [N] + [M]*nonlin_layers + [M]
    keys = jrandom.split(key, len(layer_sizes)-1)
    if repeat_layers:
        _keys = keys
        keys = jnp.repeat(_keys[0][None], _keys.shape[0], 0)
        if N != M:
            # REPLACED!!!!!!!!!
            keys = keys.at[1:].set(_keys[-1])
            # keys = index_update(keys, index[1:], _keys[-1])
    return [unif_nica_layer(n, m, k) for (n, m, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def init_encoder_params(x_dim, s_dim, hidden_dim, hidden_layers, key):
    '''BEWARE: Assumes factorized distribution
        and equal width in all hidden layers'''
    layer_sizes = [x_dim] + [hidden_dim]*hidden_layers + [s_dim*2]
    keys = jrandom.split(key, len(layer_sizes)-1)
    return [init_layer_params(m, n, k) for (m, n, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def init_decoder_params(x_dim, s_dim, hidden_dim, hidden_layers, key):
    '''BEWARE: Assumes equal width in all hidden layers'''
    layer_sizes = [s_dim] + [hidden_dim]*hidden_layers + [x_dim]
    keys = jrandom.split(key, len(layer_sizes)-1)
    return [init_layer_params(m, n, k) for (m, n, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def encoder_mlp(params, x, activation='xtanh', slope=0.1):
    """Forward pass for encoder MLP that predicts likelihood natparams.
    Args:
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs p(x|s) natparams (v, W) -- see derivation
    """
    if activation == 'xtanh':
        act = xtanh(slope)
    else:
        act = SmoothLeakyRelu(slope)
        #act = lambda x: nn.leaky_relu(x, slope)
    z = x
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            W, b = hidden_params[i]
            z = act(z@W + b)
    final_W, final_b = params[-1]
    z = z@final_W + final_b
    v, W_diag = jnp.split(z, 2)
    W = -jnp.diag(nn.softplus(W_diag))
    return v, W


def decoder_mlp(params, x, activation='xtanh', slope=0.1):
    """Forward pass for encoder MLP for estimating nonlinear mixing function.
    Args: (IGNORE; OLD)
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs f(s)
    """
    if activation == 'xtanh':
        act = xtanh(slope)
    else:
        act = SmoothLeakyRelu(slope)
        #act = lambda x: nn.leaky_relu(x, slope)
    z = x
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            W, b = hidden_params[i]
            z = act(z@W + b)
    final_W, final_b = params[-1]
    z = z@final_W + final_b
    return z


def nica_mlp(params, s, activation='xtanh', slope=0.1):
    """Forward pass for encoder MLP for estimating nonlinear mixing function.
    Args: (OLD; IGNORE)
        params (list): nested list where each element is a list of weight
            matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
        inputs (matrix): input data.
        slope (float): slope to control the nonlinearity of the activation
            function.
    Returns:
        Outputs f(s)
    """
    if activation == 'xtanh':
        act = xtanh(slope)
    else:
        act = SmoothLeakyRelu(slope)
    z = s
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            z = act(z@hidden_params[i])
    A_final = params[-1]
    z = z@A_final
    return z


if __name__ == "__main__":

    key = jrandom.PRNGKey(0)
    x_dim = 10
    s_dim = 4
    hidden_dim = 20
    hidden_layers = 5
    s = jnp.ones((s_dim,))

    # nonlinear ICA fwd test
    nica_params = init_nica_params(s_dim, x_dim, 3, key, repeat_layers=False)
    x = nica_mlp(nica_params, s, slope=0.1)
    dp = init_decoder_params(x_dim, s_dim, 32, 1, key)
    decoder_mlp(dp, s)

    params = init_encoder_params(x_dim, s_dim, hidden_dim,
                                 hidden_layers, key)
    out = encoder_mlp(params, x)

    pdb.set_trace()

    # linear ICA fwd test
    key = jrandom.PRNGKey(1)
    s_dim = 10
    x_dim = 10
    ica_params = init_nica_params(s_dim, x_dim, 0, key, repeat_layers=False)
    unif_nica_layer(4, 5, key, iter_4_cond=1e3)


def create_encoder_train_state(rng, config):
  """Creates initial `TrainState`."""
  cnn = encoder_CNN(c_out=config.n)
  params = cnn.init(rng, jnp.ones([32, 32, 3]))['params']
  # tx = optax.sgd(config.nn_learning_rate, config.momentum)
  tx = optax.adam(config.nn_learning_rate)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)

def create_decoder_train_state(rng, config):
  """Creates initial `TrainState`."""
  cnn = decoder_CNN(c_out=3,c_input=config.n)
  params = cnn.init(rng, jnp.ones(config.n,))['params']
  # tx = optax.sgd(config.nn_learning_rate, config.momentum)
  tx = optax.adam(config.nn_learning_rate)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)


# class encoder_CNN(fnn.Module):
#   """A simple CNN model."""
#   c_out: int
#
#   @fnn.compact
#   def __call__(self, x):
#     x = fnn.Conv(features=32, kernel_size=(3, 3))(x)
#     x = fnn.relu(x)
#     x = fnn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     x = fnn.Conv(features=64, kernel_size=(3, 3))(x)
#     x = fnn.relu(x)
#     x = fnn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     # x = x.reshape((x.shape[0], -1))  # flatten
#     x = x.ravel()
#     x = fnn.Dense(features=256)(x)
#     x = fnn.relu(x)
#     x = fnn.Dense(features=2*self.c_out)(x)
#     v, W_diag = jnp.split(x.reshape(2*self.c_out), 2)
#     W = -jnp.diag(nn.softplus(W_diag))
#     return v, W

class encoder_CNN(fnn.Module):
  """A simple CNN model."""
  c_out: int

  @fnn.compact
  def __call__(self, x):
    x = fnn.Conv(features=5, kernel_size=(5, 5))(x)
    x = fnn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = fnn.relu(x)
    x = fnn.Conv(features=8, kernel_size=(5, 5))(x)
    x = fnn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = fnn.relu(x)
    # x = x.reshape((x.shape[0], -1))  # flatten
    x = x.ravel()
    x = fnn.Dense(features=20)(x)
    x = fnn.relu(x)
    x = fnn.Dense(features=2*self.c_out)(x)
    v, W_diag = jnp.split(x.reshape(2*self.c_out), 2)
    W = -jnp.diag(nn.softplus(W_diag))
    return v, W

# class decoder_CNN(fnn.Module):
#     c_out : int
#     c_input : int
#
#     @fnn.compact
#     def __call__(self, x):
#         x = fnn.Dense(features=2*49*self.c_input)(x)
#         x = fnn.gelu(x)
#         x = x.reshape(7, 7, -1)
#         x = fnn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
#         x = fnn.gelu(x)
#         x = fnn.Conv(features=64, kernel_size=(3, 3))(x)
#         x = fnn.gelu(x)
#         x = fnn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
#         x = fnn.gelu(x)
#         x = fnn.Conv(features=64, kernel_size=(3, 3))(x)
#         x = fnn.gelu(x)
#         x = fnn.ConvTranspose(features=self.c_out, kernel_size=(3, 3))(x)
#         x = fnn.tanh(x)
#         return x

class decoder_CNN(fnn.Module):
    c_out : int
    c_input : int

    @fnn.compact
    def __call__(self, x):
        x = fnn.Dense(features=2*64*self.c_input)(x)
        x = fnn.gelu(x)
        x = x.reshape(8, 8, -1)
        x = fnn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = fnn.gelu(x)
        x = fnn.Conv(features=32, kernel_size=(3, 3))(x)
        x = fnn.gelu(x)
        x = fnn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = fnn.gelu(x)
        x = fnn.Conv(features=32, kernel_size=(3, 3))(x)
        x = fnn.gelu(x)
        x = fnn.ConvTranspose(features=self.c_out, kernel_size=(3, 3))(x)
        x = fnn.tanh(x)
        print('decoder x shape',x.shape)
        return x




@jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images)
    one_hot = fnn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)