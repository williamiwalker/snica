from functools import partial
import pdb

import jax
from jax import jit, vmap
from jax.config import config
from jax.experimental import host_callback
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.stats import multivariate_normal as gaussian
from numpy import newaxis
import numpy as np

from func_estimators import decoder_mlp, encoder_mlp
from inference import make_inference
from utils import (
    gaussian_sample_from_mu_prec,
    get_hmm_natparams,
    get_prec_mat,
    get_prior_natparams,
    get_rhos,
    get_transition_natparams,
    invmp,
)

config.update("jax_enable_x64", True)


#@jit
def sample_fwd_step(z_prev, pw_posterior_with_key):
    qzlag_z, key = pw_posterior_with_key
    mu_pw, prec_pw = qzlag_z
    d = z_prev.shape[0]
    mu_cond = mu_pw[d:]-invmp(prec_pw[d:, d:],
                              prec_pw[d:, :d]@(z_prev-mu_pw[:d]))
    prec_cond = prec_pw[d:, d:]
    z_new = gaussian_sample_from_mu_prec(mu_cond, prec_cond, key)
    return z_new, z_new


#@jit
def sample_forward(lds_posteriors, key):
    qz, qzlag_z = lds_posteriors
    mu, prec = qz
    keys = jrandom.split(key, mu.shape[0])
    z_prior = gaussian_sample_from_mu_prec(mu[0], prec[0], keys[0])
    z = scan(sample_fwd_step, z_prior, (qzlag_z, keys[1:]))[1]
    return jnp.vstack((z_prior, z))


#@jit
def gaussian_entropy(prec):
    d = prec.shape[0]
    V = invmp(prec, jnp.eye(prec.shape[0]))
    return 0.5*(jnp.linalg.slogdet(V)[1]+d*jnp.log(2*jnp.pi*jnp.e))


#@jit
def lds_entropy(qz_prec, qzlag_z_prec):
    marginal_entropies = vmap(lambda a:
                              vmap(gaussian_entropy)(a[1:-1]))(qz_prec)
    pw_entropies = vmap(vmap(gaussian_entropy))(qzlag_z_prec)
    return jnp.sum(pw_entropies)-jnp.sum(marginal_entropies)


#@jit
def E_logp_lds(prior_natparams, transition_natparams, qz, qzlag_z, qu):
    rho = vmap(get_rhos)((prior_natparams, transition_natparams),
                         (qz, qzlag_z))
    E_logp_z = jnp.sum(rho*jnp.exp(qu))
    return E_logp_z


#@jit
def KL_qp_u(hmm_natparams, qu, quu):
    pw_part = jnp.sum((quu-hmm_natparams[1][:, None, :, :])*jnp.exp(quu))
    marg_part = jnp.sum(jnp.exp(qu[:, 1:-1, :])*qu[:, 1:-1, :])+jnp.sum(
        jnp.exp(qu[:, 0, :])*hmm_natparams[0])
    return pw_part-marg_part


#@jit
def E_sampling_likelihood(x, s_sample, theta, R):
    x_sample_mu = vmap(vmap(lambda a: decoder_mlp(theta, a),
                            in_axes=-1, out_axes=-1))(s_sample)
    # jax.debug.print('x_sample_mu min {x} max {y}', x=jnp.min(x_sample_mu), y=jnp.max(x_sample_mu))
    V = invmp(R, jnp.eye(R.shape[0]))
    # print('V shape', V.shape,x_sample_mu.shape,x.shape)
    E_logp_xs = vmap(lambda a: vmap(gaussian.logpdf, in_axes=(-1, -1, None),
                     out_axes=-1)(x, a, V))(x_sample_mu)
    # print('E logp xs shape', E_logp_xs.shape)
    return E_logp_xs.sum(1).mean()

#@jit
def E_sampling_likelihood_MNIST(x, s_sample, decoder_state, dec_params, R):
    # x_sample_mu = vmap(vmap(lambda a: decoder_mlp(theta, a),
    #                         in_axes=-1, out_axes=-1))(s_sample)
    x_sample_mu = vmap(vmap(lambda a: decoder_state.apply_fn({'params':dec_params}, a),
                            in_axes=0, out_axes=-1))(s_sample) # TODO: MAKE SURE YOU ARE OUTPUTING LIKELIHOOD NAT PARAMS
    # jax.debug.print('x_sample_mu min {x} max {y}', x=jnp.min(x_sample_mu), y=jnp.max(x_sample_mu))
    # jax.debug.print('x min {x} max {y}', x=jnp.min(x), y=jnp.max(x))
    # print('x sample mu shape',x_sample_mu.shape,x_sample_mu.reshape(x.shape[0],-1).shape, x.reshape(x.shape[0],-1).shape)
    V = invmp(R, jnp.eye(R.shape[0]))
    # print('V shape',V.shape)
    # print('gaussian logpdf', gaussian.logpdf(x.reshape(x.shape[0],-1)[0], x_sample_mu.reshape(x.shape[0],-1)[0], V).block_until_ready())
    print('x shape',x.shape, x_sample_mu.shape, V.shape)
    E_logp_xs = vmap(gaussian.logpdf, in_axes=(0, 0, None),
                     out_axes=-1)(x.reshape(x.shape[0],-1), x_sample_mu.reshape(x.shape[0],-1), V)
    # print('E logp xs shape', E_logp_xs.shape)
    # E_logp_xs = vmap(lambda a: vmap(gaussian.logpdf, in_axes=(0, 0, None),
    #                  out_axes=-1)(x.reshape(x.shape[0],-1), a, V))(x_sample_mu.reshape(x.shape[0],-1))
    return E_logp_xs.sum()

@partial(jit, static_argnums=(8, 9,))
def ELBO(x, R, lds_params, log_hmm_params, phi, theta, nu, key,
         inference_iters, num_samples):
    '''

    :param x: observations
    :param R:
    :param lds_params:
    :param log_hmm_params:
    :param phi: encoder NN params
    :param theta: decoder NN params
    :param nu: burn in/ annealing
    :param key:
    :param inference_iters:
    :param num_samples:
    :return:
    '''
    # transform into natural parameter form
    M, T = x.shape
    N, K = log_hmm_params[0].shape
    d = lds_params[0].shape[-1]
    hmm_params = jax.tree_map(lambda a: jnp.exp(a), log_hmm_params)
    # ensure probabilities are normalized
    hmm_params = jax.tree_map(lambda a: a/jnp.sum(a, -1, keepdims=True),
                              hmm_params)
    # transform into natparams
    hmm_natparams = vmap(get_hmm_natparams)(hmm_params)
    prior_natparams = vmap(vmap(get_prior_natparams))(lds_params[:2])
    transition_natparams = vmap(vmap(get_transition_natparams))(lds_params[2:])
    if len(phi) > 1:
        likelihood_natparams = vmap(lambda a: encoder_mlp(phi, a),
                                    in_axes=-1, out_axes=(-1, -1))(x)
    elif len(phi) == 1:
        C = theta[0][0].T
        print('C.shape',C.T.shape, R.shape, x.shape)
        np0 = C.T@R@x
        np1 = -0.5*C.T@R@C
        likelihood_natparams = (np0, jnp.repeat(np1[:, :, jnp.newaxis],
                                                x.shape[1], 2))

    # print('all shapes',x.shape, likelihood_natparams[0].shape, likelihood_natparams[1].shape, np.max(likelihood_natparams[0]), np.min(likelihood_natparams[0]), np.max(likelihood_natparams[1]), np.min(likelihood_natparams[1]))
    jax.debug.print('likelihood nat params 0 min {x} max {y}',x=jnp.min(likelihood_natparams[0]),y=jnp.max(likelihood_natparams[0]))
    jax.debug.print('likelihood nat params 1 min {x} max {y}', x=jnp.min(likelihood_natparams[1]), y=jnp.max(likelihood_natparams[1]))
    jax.debug.print('liklihood params {x}', x=likelihood_natparams)

    # random initialization
    key, qkey = jrandom.split(key)
    qkeys = jrandom.split(qkey, 6)

    qu = jax.tree_map(lambda a: jnp.log(a/a.sum(-1, keepdims=True)),
                      jrandom.uniform(qkeys[0], (N, T, K)))
    quu = jax.tree_map(lambda a: jnp.log(a/a.sum((-2, -1), keepdims=True)),
                       jrandom.uniform(qkeys[1], (N, T-1, K, K)))
    qz_mu = jrandom.normal(qkeys[2], (N, T, d))
    qzlagz_mu = jrandom.normal(qkeys[3], (N, T-1, 2*d))
    qz_prec = vmap(lambda k: get_prec_mat(d, 1., k)*jnp.eye(d))(
        jrandom.split(qkeys[4], N*T)).reshape((N, T, d, d))
    qzlagz_prec = vmap(lambda k: get_prec_mat(2*d, 1., k)*jnp.eye(2*d))(
        jrandom.split(qkeys[5], N*(T-1))).reshape((N, T-1, 2*d, 2*d))
    qz = (qz_mu, qz_prec)
    qzlag_z = (qzlagz_mu, qzlagz_prec)

    # run inference
    inference_runner = make_inference(hmm_natparams, prior_natparams,
                                      transition_natparams,
                                      likelihood_natparams)
    qz, qzlag_z, qu, quu = scan(inference_runner, (qz, qzlag_z, qu, quu),
                                jnp.arange(inference_iters))[0]

    # sample
    key, samplekey = jrandom.split(key)
    z_sample = vmap(
        lambda k: vmap(sample_forward)((qz, qzlag_z), jrandom.split(k, N))
    )(jrandom.split(samplekey, num_samples))
    print('z sample shape', z_sample.shape)
    s_sample = z_sample[:, :, :, 0]
    print('s sample shape', s_sample.shape)

    # compute elbo
    E_logp_xs = E_sampling_likelihood(x, s_sample, theta, R)
    E_logp_z = E_logp_lds(prior_natparams, transition_natparams,
                          qz, qzlag_z, qu)
    H_z = lds_entropy(qz[1], qzlag_z[1])
    KL_u = KL_qp_u(hmm_natparams, qu, quu)
    elbo = E_logp_xs+nu*(E_logp_z+H_z-KL_u)
    KL = -H_z+KL_u-E_logp_z
    #host_callback.id_print({'KL': KL, 'Elxs':E_logp_xs, 'Elz': E_logp_z,
    #                        'H_z': H_z, 'negKLu': -KL_u})
    return elbo, (qz, qzlag_z, qu, quu)


#@partial(jit, static_argnums=(7, 8,))
def avg_neg_ELBO(x, mix_params, lds_params, hmm_params, phi, theta, nu,
                 key, inference_iters, num_samples, minibatch=False):
    if minibatch:
        keys = jrandom.split(key, x.shape[0])
        elbo, posteriors = vmap(
            lambda a, b: ELBO(a, mix_params, lds_params, hmm_params,
                              phi, theta, nu, b, inference_iters, num_samples)
        )(x, keys)
        elbo = elbo.mean()
    else:
        elbo, posteriors = ELBO(x, mix_params, lds_params, hmm_params, phi,
                                theta, nu, key, inference_iters, num_samples)
    return -elbo, posteriors


#@partial(jit, static_argnums=(7, 8,))
def avg_neg_ELBO_MNIST(x, mix_params, lds_params, hmm_params, encoder_state, enc_params, decoder_state, dec_params, nu,
                 key, inference_iters, num_samples, minibatch=False):
    if minibatch:
        keys = jrandom.split(key, x.shape[0])
        elbo, posteriors = vmap(
            lambda a, b: ELBO_MNIST(a, mix_params, lds_params, hmm_params,
                                    encoder_state, enc_params, decoder_state, dec_params, nu, b, inference_iters, num_samples)
        )(x, keys)
        elbo = elbo.mean()
    else:
        elbo, posteriors = ELBO_MNIST(x, mix_params, lds_params, hmm_params, encoder_state, enc_params, decoder_state,
                                      dec_params, nu, key, inference_iters, num_samples)
    return -elbo, posteriors

@partial(jit, static_argnums=(10, 11,))
def ELBO_MNIST(x, R, lds_params, log_hmm_params, encoder_state, enc_params, decoder_state, dec_params, nu, key,
         inference_iters, num_samples):
    '''

    :param x: observations
    :param R: variance of noise on output of decoder
    :param lds_params:
    :param log_hmm_params:
    :param phi: encoder NN params
    :param theta: decoder NN params
    :param nu: burn in/ annealing
    :param key:
    :param inference_iters:
    :param num_samples:
    :return:
    '''
    # transform into natural parameter form
    # M, T = x.shape
    T = x.shape[0]
    N, K = log_hmm_params[0].shape
    d = lds_params[0].shape[-1]
    hmm_params = jax.tree_map(lambda a: jnp.exp(a), log_hmm_params)
    # ensure probabilities are normalized
    hmm_params = jax.tree_map(lambda a: a/jnp.sum(a, -1, keepdims=True),
                              hmm_params)
    # transform into natparams
    hmm_natparams = vmap(get_hmm_natparams)(hmm_params)
    prior_natparams = vmap(vmap(get_prior_natparams))(lds_params[:2])
    transition_natparams = vmap(vmap(get_transition_natparams))(lds_params[2:])
    # if len(phi) > 1:
    #     likelihood_natparams = vmap(lambda a: encoder_mlp(phi, a),
    #                                 in_axes=-1, out_axes=(-1, -1))(x)
    likelihood_natparams = vmap(lambda a: encoder_state.apply_fn({'params':enc_params}, a),
                                in_axes=0, out_axes=(-1, -1))(x) # TODO: MAKE SURE YOU ARE OUTPUTTING LIKELIHOOD NAT PARAMS

    # print('all shapes',x.shape, likelihood_natparams[0].shape, likelihood_natparams[1].shape, np.max(likelihood_natparams[0]), np.min(likelihood_natparams[0]), np.max(likelihood_natparams[1]), np.min(likelihood_natparams[1]))
    # jax.debug.print('likelihood nat params 0 min {x} max {y}',x=jnp.min(likelihood_natparams[0]),y=jnp.max(likelihood_natparams[0]))
    # jax.debug.print('likelihood nat params 1 min {x} max {y}', x=jnp.min(likelihood_natparams[1]), y=jnp.max(likelihood_natparams[1]))
    # jax.debug.print('liklihood params {x}', x=likelihood_natparams)
    # jax.experimental.host_callback.id_print(jnp.min(likelihood_natparams[0]))
    # elif len(phi) == 1:
    #     C = theta[0][0].T
    #     print('C.shape',C.T.shape, R.shape, x.shape)
    #     np0 = C.T@R@x
    #     np1 = -0.5*C.T@R@C
    #     likelihood_natparams = (np0, jnp.repeat(np1[:, :, jnp.newaxis],
    #                                             x.shape[1], 2))

    # random initialization
    key, qkey = jrandom.split(key)
    qkeys = jrandom.split(qkey, 6)

    qu = jax.tree_map(lambda a: jnp.log(a/a.sum(-1, keepdims=True)),
                      jrandom.uniform(qkeys[0], (N, T, K)))
    quu = jax.tree_map(lambda a: jnp.log(a/a.sum((-2, -1), keepdims=True)),
                       jrandom.uniform(qkeys[1], (N, T-1, K, K)))
    qz_mu = jrandom.normal(qkeys[2], (N, T, d))
    qzlagz_mu = jrandom.normal(qkeys[3], (N, T-1, 2*d))
    qz_prec = vmap(lambda k: get_prec_mat(d, 1., k)*jnp.eye(d))(
        jrandom.split(qkeys[4], N*T)).reshape((N, T, d, d))
    qzlagz_prec = vmap(lambda k: get_prec_mat(2*d, 1., k)*jnp.eye(2*d))(
        jrandom.split(qkeys[5], N*(T-1))).reshape((N, T-1, 2*d, 2*d))
    qz = (qz_mu, qz_prec)
    qzlag_z = (qzlagz_mu, qzlagz_prec)

    # run inference
    inference_runner = make_inference(hmm_natparams, prior_natparams,
                                      transition_natparams,
                                      likelihood_natparams)
    qz, qzlag_z, qu, quu = scan(inference_runner, (qz, qzlag_z, qu, quu),
                                jnp.arange(inference_iters))[0]

    # sample
    key, samplekey = jrandom.split(key)
    z_sample = vmap(
        lambda k: vmap(sample_forward)((qz, qzlag_z), jrandom.split(k, N))
    )(jrandom.split(samplekey, num_samples))
    # print('z sample shape', z_sample.shape)
    s_sample = jnp.transpose(z_sample[:, :, :, 0],(2,0,1))
    # print('s sample shape', s_sample.shape)

    # compute elbo
    E_logp_xs = E_sampling_likelihood_MNIST(x, s_sample, decoder_state, dec_params, R)
    E_logp_z = E_logp_lds(prior_natparams, transition_natparams,
                          qz, qzlag_z, qu)
    H_z = lds_entropy(qz[1], qzlag_z[1])
    KL_u = KL_qp_u(hmm_natparams, qu, quu)
    elbo = E_logp_xs+nu*(E_logp_z+H_z-KL_u)
    KL = -H_z+KL_u-E_logp_z
    #host_callback.id_print({'KL': KL, 'Elxs':E_logp_xs, 'Elz': E_logp_z,
    #                        'H_z': H_z, 'negKLu': -KL_u})
    return elbo, (qz, qzlag_z, qu, quu)


@jit
def apply_NN(state, images):
    """Computes gradients, loss and accuracy for a single batch."""
    return state.apply_fn({'params': state.params}, images)