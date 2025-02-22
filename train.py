import os
import time
import pdb

from jax.config import config
config.update("jax_debug_nans", True)

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jax import vmap, jit, value_and_grad
from jax.lax import cond
from optax import chain, piecewise_constant_schedule, scale_by_schedule

from functools import partial
from elbo import avg_neg_ELBO, avg_neg_ELBO_MNIST
from func_estimators import init_encoder_params, init_decoder_params
from func_estimators import decoder_mlp, encoder_CNN, decoder_CNN, create_encoder_train_state, create_decoder_train_state
from utils import matching_sources_corr, plot_ic, plot_states_learned, best_prediction
from utils import nsym_grad, sym_grad, get_prec_mat, plot_loss
from utils import save_best, load_best_ckpt, save_best_HMM, load_best_ckpt_HMM, save_best_qu

import matplotlib.pyplot as plt
import numpy as np

from flax.training import train_state

# def create_train_state(rng, config):
#   """Creates initial `TrainState`."""
#   cnn = CNN()
#   params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
#   tx = optax.sgd(config.learning_rate, config.momentum)
#   return train_state.TrainState.create(
#       apply_fn=cnn.apply, params=params, tx=tx)

def full_train_MNIST(obs, states, args, est_key):
    print("Running with:", args)
    # unpack some of the args
    # N = z_mu.shape[0] # reinitiate this!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # M, T = x.shape # reinitiate this!!!!!!!!!!!!!!!!!!!!!!!!!!!
    num_epochs = args.num_epochs
    inference_iters = args.inference_iters
    num_samples = args.num_samples
    # enc_hidden_units = args.hidden_units_enc
    # dec_hidden_units = args.hidden_units_dec
    # enc_hidden_layers = args.hidden_layers_enc
    # dec_hidden_layers = args.hidden_layers_dec
    lr_nn = args.nn_learning_rate
    lr_gm = args.gm_learning_rate
    decay_interval = args.decay_interval
    decay_rate = args.decay_rate
    burnin_len = args.burnin
    plot_freq = args.plot_freq
    # mix_params, lds_params, hmm_params = params
    # _, K, d = lds_params[0].shape
    K = args.k
    d = args.d
    N = args.n
    T = args.t
    batch_size = args.batch_size ##########ADD
    save_folder = args.saveFolder

    # initialize pgm parameters randomly
    est_key = jrandom.PRNGKey(est_key)
    est_key, Rkey = jrandom.split(est_key)
    est_key, *hmmkeys = jrandom.split(est_key, 3)
    est_key, *ldskeys = jrandom.split(est_key, 6)

    for batch in obs:
        fbatch = batch
        break
    # print('fbatch shape',fbatch.reshape(fbatch.shape[0] * fbatch.shape[1],-1).shape)
    # print('cov',jnp.cov(fbatch.reshape(-1,fbatch.shape[0] * fbatch.shape[1])).shape)
    x_vars = jnp.diag(jnp.cov(fbatch.reshape(-1,fbatch.shape[0] * fbatch.shape[1]))) # observations used to initiate variables!!!!!!!!!!!!!!!!!!!!!
    M = len(x_vars)
    R_est = jnp.linalg.inv(jnp.diag( # observations used to initiate variables!!!!!!!!!!!!!!!!!!!!!
        jrandom.uniform(Rkey, (M,), minval=0.1*jnp.min(x_vars),
                        maxval=0.5*jnp.max(x_vars))))
    # R_est = 1.0/jnp.diag( # observations used to initiate variables!!!!!!!!!!!!!!!!!!!!!
    #     jrandom.uniform(Rkey, (M,), minval=0.1*jnp.min(x_vars), maxval=0.5*jnp.max(x_vars)))
    # print('R_est shape',R_est.shape)
    hmm_est = jax.tree_map(
        lambda a: jnp.log(a / a.sum(-1, keepdims=True)),
        (jrandom.uniform(hmmkeys[0], (N, K)),
         jrandom.uniform(hmmkeys[1], (N, K, K)))
    )
    b_prior_est = jrandom.uniform(ldskeys[0], (N, K, d), minval=-1, maxval=1)
    b_est = jrandom.uniform(ldskeys[1], (N, K, d), minval=-1, maxval=1)
    # B_est = jrandom.uniform(ldskeys[2], (N, K, d, d), minval=-1, maxval=1)
    print('INIT B_est TO BE ALL ZEROS')
    B_est = jnp.zeros((N, K, d, d))
    Q_prior_est = vmap(lambda k: get_prec_mat(d, 10., k)*jnp.eye(d))(
        jrandom.split(ldskeys[3], N*K)).reshape((N, K, d, d))
    Q_est = vmap(lambda k: get_prec_mat(d, 10., k)*jnp.eye(d))(
        jrandom.split(ldskeys[4], N*K)).reshape((N, K, d, d))
    lds_est = (b_prior_est, Q_prior_est, B_est, b_est, Q_est)

    # # for debugging at ground truth pgm parameters
    # if args.gt_gm_params:
    #     R_est = mix_params[1]
    #     lds_est = lds_params
    #     hmm_est = jax.tree_map(lambda a: jnp.log(a), hmm_params)

    # initialize func estimators
    key, enc_key, dec_key = jrandom.split(est_key, 3)
    # theta = init_decoder_params(M, N, dec_hidden_units,
    #                             dec_hidden_layers, dec_key)
    # Initialize parameters of decoder with random key and latents
    decoder_state = create_decoder_train_state(enc_key, args)
    dec_params = decoder_state.params

    # note that encoder is only needed in nonlinear (our default) case...
    # phi = init_encoder_params(M, N, enc_hidden_units,
    #                               enc_hidden_layers, enc_key)
    encoder_state = create_encoder_train_state(enc_key, args)
    enc_params = encoder_state.params


    # initialize training
    gm_params = (R_est, lds_est, hmm_est)
    # nn_params = (phi, theta)
    all_params = gm_params
    param_labels = ('gm')
    schedule_fn = piecewise_constant_schedule(1., {decay_interval: decay_rate})
    tx = optax.multi_transform({
        'gm': chain(optax.adam(lr_gm), scale_by_schedule(schedule_fn)),
        'nn': chain(optax.adam(lr_nn), scale_by_schedule(schedule_fn))},
        param_labels)
    opt_state = tx.init(all_params)
    start_epoch = 0

    # option to resume to checkpoint
    if args.resume_best:
        start_epoch, all_params, opt_state, tx, enc_params, enc_tx, dec_params, dec_tx = load_best_ckpt_HMM(args.saveFolder)
        encoder_state.params = enc_params
        decoder_state.params = dec_params
        encoder_state.tx = enc_tx
        decoder_state.tx = dec_tx

    # define training step
    @partial(jit, static_argnums=(8, 9))
    def training_step(epoch_num, params, opt_state, encoder_state, enc_params, decoder_state, dec_params, obs_batch, # observations used in training!!!!!!!!!!!!!!!!!!!!!
                      inference_iters, num_samples, burnin, key):
        """Performs gradient step on the function estimator
               MLP parameters on the ELBO.
        """

        # symmetrization of precision matrix grads - can also use nsym_grad()
        def sym_diag_grads(mat):
            return sym_grad(mat) * jnp.eye(mat.shape[0])


        # unpack
        key, subkey = jrandom.split(key)
        # key, subkey_batch = jrandom.split(key)
        R_est, lds_est, hmm_est = params
        # enc_params = encoder_state.params
        # dec_params = decoder_state.params

        # option to anneal elbo KL terms by factor
        nu = cond(burnin > 0,
                  lambda _: jnp.clip(epoch_num/(burnin+1e-5), a_max=1.0),
                  lambda _: 1., burnin)

        # get gradients
        # # create batches
        # train_ds_size = len(obs)
        # steps_per_epoch = train_ds_size // batch_size
        #
        # perms = jax.random.permutation(subkey_batch, len(obs))
        # perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        # perms = perms.reshape((steps_per_epoch, batch_size))
        #
        # epoch_elbo = [] # keep track of ELBO of each batch
        #
        # # for each batch
        # for perm in perms:
        #     batch_images = x[perm, ...]
        (n_elbo, posteriors), g = value_and_grad(
            avg_neg_ELBO_MNIST, argnums=(1, 2, 3, 5, 7,), has_aux=True)(
                obs_batch, R_est, lds_est, hmm_est, encoder_state, enc_params, decoder_state, dec_params, nu, # observations used to evaluate loss (ELBO)!!!!!!!!!!!!!!!!!!!!!
                subkey, inference_iters, num_samples,
        )
        # epoch_elbo.append(n_elbo) # keep track of ELBO of each batch

        # unpack grads
        R_g, lds_g, hmm_g, enc_g, dec_g = g
        b_prior_g, Q_prior_g, B_g, b_g, Q_g = lds_g
        pi_g, A_g = hmm_g
        print('SET B_est GRAD TO ZERO')
        B_g = jnp.zeros((N, K, d, d))

        # symmetrization of precision matrix grads - can also use nsym_grad()
        R_g = sym_diag_grads(R_g)
        Q_prior_g = vmap(vmap(sym_diag_grads))(Q_prior_g)
        Q_g = vmap(vmap(sym_diag_grads))(Q_g)

        # pack up
        lds_g = (b_prior_g, Q_prior_g, B_g, b_g, Q_g)
        hmm_g = (pi_g, A_g)
        gm_g = (R_g, lds_g, hmm_g)
        # nn_g = (enc_g, dec_g)
        g = gm_g

        # perform gradient updates
        # TODO: FIX opt_state
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)

        # update encoder and decoder params
        encoder_state = encoder_state.apply_gradients(grads=enc_g)
        decoder_state = decoder_state.apply_gradients(grads=dec_g)

        # train_elbo = np.mean(epoch_elbo)

        return n_elbo, posteriors, params, opt_state, encoder_state, decoder_state



    @partial(jit, static_argnums=(5, 6))
    def infer_step(epoch_num, params, encoder_state, decoder_state, obs_batch, # observations used in inference!!!!!!!!!!!!!!!!!!!!!
                   inference_iters, num_samples, burnin, key):
        """Perform inference without gradient step for eval purposes
               MLP parameters on the ELBO.
        """
        # unpack
        key, subkey = jrandom.split(key)
        R_est, lds_est, hmm_est = params[0]
        enc_params = encoder_state.params
        dec_params = decoder_state.params

        # always turn annealing off for eval
        nu = 1.

        # inference step
        n_elbo, posteriors = avg_neg_ELBO_MNIST(obs_batch, R_est, lds_est, hmm_est, encoder_state, enc_params, decoder_state, dec_params, nu,
                # observations used to evaluate loss (ELBO)!!!!!!!!!!!!!!!!!!!!!
                subkey, inference_iters, num_samples,
            )
        return n_elbo, posteriors


    # set plot
    fig, ax = plt.subplots(2, figsize=(10 * N, 6),
                           gridspec_kw={'height_ratios': [1, 2]})
    ax2 = ax
    # for n in range(N):
    #     print('shape check',n, ax2.shape, ax.shape)
    #     ax2[1, n] = ax[1, n].twinx()

    # train
    best_elbo = -jnp.inf
    loss_track = np.zeros(num_epochs)
    for epoch in range(start_epoch, num_epochs):
        tic = time.time()
        niters = min(inference_iters, ((epoch // 100) + 1) * 5)
        key, trainkey = jrandom.split(key, 2)
        batch_losses = np.zeros(args.num_sequences)
        for indb, batch in enumerate(obs):

            if args.eval_only:
                # infer posteriors for evaluation without grad step
                n_elbo, posteriors = infer_step(
                    epoch, all_params, encoder_state, decoder_state, batch[0], niters, # observations used in inference!!!!!!!!!!!!!!!!!!!!!
                    num_samples, burnin_len, trainkey)
            else:
                # training step
                n_elbo, posteriors, all_params, opt_state, encoder_state, decoder_state = training_step(
                    epoch, all_params, opt_state, encoder_state, enc_params, decoder_state, dec_params, batch[0], niters, # observations used in training!!!!!!!!!!!!!!!!!!!!!
                    num_samples, burnin_len, trainkey)
                enc_params = encoder_state.params
                dec_params = decoder_state.params

            print('batch loss: ',indb,' ',-n_elbo)
            batch_losses[indb] = -n_elbo

        loss_track[epoch] = batch_losses.mean()

        # TODO: FIX EVALUATION
        # evaluate
        # qz, qzlag_z, qu, quu = posteriors
        # mcc, _, sort_idx = matching_sources_corr(qz[0][:, :, 0], z_mu[:, :, 0])
        # f_mu_est = vmap(decoder_mlp, in_axes=(None, -1),
        #                 out_axes=-1)(all_params[1][1], qz[0][:, :, 0])
        # denoise_mcc = jnp.abs(jnp.diag(
        #     jnp.corrcoef(f_mu_est, f)[:M, M:])).mean()

        print("*Epoch: [{0}/{1}]\t"
              "ELBO: {2}\t"
              "num. infernce iters: {3}".format(epoch, num_epochs, -n_elbo, niters))

        if epoch % plot_freq == 0 or args.eval_only:
            # plot
            qz, qzlag_z, qu, quu = posteriors
            # Get Best State Permutation
            perm = best_prediction(np.array(qu[0]), np.array(states[-1]))
            plot_start = int(T/2)
            plot_len = 500
            plot_end = plot_start+plot_len
            for n in range(N):
                # print('qz shape', qz[0].shape, qz[1].shape)
                qz_mu_n = qz[0][perm][n][plot_start:plot_end]
                qz_prec_n = qz[1][perm][n][plot_start:plot_end]
                qu_n = jnp.exp(qu[perm][n][plot_start:plot_end])
                u_n = states[-1][plot_start:plot_end]
                # print('states and u_n ',plot_start,plot_end,states.shape)
                # z_mu_n = z_mu[n][plot_start:plot_end]
                # plot_ic(u_n, z_mu_n, qu_n, qz_mu_n, qz_prec_n,
                #         ax[0, n], ax[1, n], ax2[1, n])
                # plot_states_learned(save_folder, u_n, qu_n, qz_mu_n, qz_prec_n,
                #         ax[0, n], ax[1, n], ax2[1, n])
                plot_states_learned(save_folder, np.array(u_n), np.array(qu_n), np.array(qz_mu_n), np.array(qz_prec_n),
                                    ax[0], ax[1], ax2[1])
                plot_loss(save_folder, loss_track)
            # plt.pause(.5)

        # saving
        if -n_elbo > best_elbo:
            best_elbo = -n_elbo
            best_params = all_params
            best_encoder_params = encoder_state.params
            best_decoder_params = decoder_state.params
            best_posters = posteriors
            save_best_HMM(save_folder, epoch, all_params, opt_state, tx, best_encoder_params, encoder_state.tx, best_decoder_params, decoder_state.tx)
            qz, qzlag_z, qu, quu = posteriors
            save_best_qu(args.saveFolder, qu)

        if args.eval_only:
            return best_params, best_posters, best_elbo
        print("Epoch took: ", time.time()-tic)
    return best_params, best_posters, best_elbo




def full_train(x, f, z, z_mu, states, params, args, est_key):
    print("Running with:", args)
    # unpack some of the args
    N = z_mu.shape[0]
    M, T = x.shape
    num_epochs = args.num_epochs
    inference_iters = args.inference_iters
    num_samples = args.num_samples
    enc_hidden_units = args.hidden_units_enc
    dec_hidden_units = args.hidden_units_dec
    enc_hidden_layers = args.hidden_layers_enc
    dec_hidden_layers = args.hidden_layers_dec
    lr_nn = args.nn_learning_rate
    lr_gm = args.gm_learning_rate
    decay_interval = args.decay_interval
    decay_rate = args.decay_rate
    burnin_len = args.burnin
    plot_freq = args.plot_freq
    mix_params, lds_params, hmm_params = params
    _, K, d = lds_params[0].shape

    # initialize pgm parameters randomly
    est_key = jrandom.PRNGKey(est_key)
    est_key, Rkey = jrandom.split(est_key)
    est_key, *hmmkeys = jrandom.split(est_key, 3)
    est_key, *ldskeys = jrandom.split(est_key, 6)
    x_vars = jnp.diag(jnp.cov(x)) # observations used to initiate variables!!!!!!!!!!!!!!!!!!!!!
    R_est = jnp.linalg.inv(jnp.diag( # observations used to initiate variables!!!!!!!!!!!!!!!!!!!!!
        jrandom.uniform(Rkey, (M,), minval=0.1*jnp.min(x_vars),
                        maxval=0.5*jnp.max(x_vars))))
    hmm_est = jax.tree_map(
        lambda a: jnp.log(a / a.sum(-1, keepdims=True)),
        (jrandom.uniform(hmmkeys[0], (N, K)),
         jrandom.uniform(hmmkeys[1], (N, K, K)))
    )
    b_prior_est = jrandom.uniform(ldskeys[0], (N, K, d), minval=-1, maxval=1)
    b_est = jrandom.uniform(ldskeys[1], (N, K, d), minval=-1, maxval=1)
    B_est = jrandom.uniform(ldskeys[2], (N, K, d, d), minval=-1, maxval=1)
    Q_prior_est = vmap(lambda k: get_prec_mat(d, 10., k)*jnp.eye(d))(
        jrandom.split(ldskeys[3], N*K)).reshape((N, K, d, d))
    Q_est = vmap(lambda k: get_prec_mat(d, 10., k)*jnp.eye(d))(
        jrandom.split(ldskeys[4], N*K)).reshape((N, K, d, d))
    lds_est = (b_prior_est, Q_prior_est, B_est, b_est, Q_est)

    # for debugging at ground truth pgm parameters
    if args.gt_gm_params:
        R_est = mix_params[1]
        lds_est = lds_params
        hmm_est = jax.tree_map(lambda a: jnp.log(a), hmm_params)

    # initialize func estimators
    key, enc_key, dec_key = jrandom.split(est_key, 3)
    theta = init_decoder_params(M, N, dec_hidden_units,
                                dec_hidden_layers, dec_key)

    # note that encoder is only needed in nonlinear (our default) case...
    if args.l > 0:
        phi = init_encoder_params(M, N, enc_hidden_units,
                                  enc_hidden_layers, enc_key)
    if args.l == 0:
        # in linear case set bias to zero (also need to transpose matrix)
        theta = [(theta[0][0], jnp.zeros(theta[0][1].shape))]
        # also in linear case set phi=theta as we need phi to run the code but
        # phi variable is not actually used
        phi = theta

    # initialize training
    gm_params = (R_est, lds_est, hmm_est)
    nn_params = (phi, theta)
    all_params = (gm_params, nn_params)
    param_labels = ('gm', 'nn')
    schedule_fn = piecewise_constant_schedule(1., {decay_interval: decay_rate})
    tx = optax.multi_transform({
        'gm': chain(optax.adam(lr_gm), scale_by_schedule(schedule_fn)),
        'nn': chain(optax.adam(lr_nn), scale_by_schedule(schedule_fn))},
        param_labels)
    opt_state = tx.init(all_params)
    start_epoch = 0

    # option to resume to checkpoint
    if args.resume_best:
        start_epoch, all_params, opt_state, tx = load_best_ckpt(args)

    # define training step
    @partial(jit, static_argnums=(4, 5))
    def training_step(epoch_num, params, opt_state, x, # observations used in training!!!!!!!!!!!!!!!!!!!!!
                      inference_iters, num_samples, burnin, key):
        """Performs gradient step on the function estimator
               MLP parameters on the ELBO.
        """
        # unpack
        key, subkey = jrandom.split(key)
        R_est, lds_est, hmm_est = params[0]
        phi, theta = params[1]

        # option to anneal elbo KL terms by factor
        nu = cond(burnin > 0,
                  lambda _: jnp.clip(epoch_num/(burnin+1e-5), a_max=1.0),
                  lambda _: 1., burnin)

        # get gradients
        (n_elbo, posteriors), g = value_and_grad(
            avg_neg_ELBO, argnums=(1, 2, 3, 4, 5,), has_aux=True)(
                x, R_est, lds_est, hmm_est, phi, theta, nu, # observations used to evaluate loss (ELBO)!!!!!!!!!!!!!!!!!!!!!
                subkey, inference_iters, num_samples,
        )

        # unpack grads
        R_g, lds_g, hmm_g, phi_g, theta_g = g
        b_prior_g, Q_prior_g, B_g, b_g, Q_g = lds_g
        pi_g, A_g = hmm_g

        # symmetrization of precision matrix grads - can also use nsym_grad()
        def sym_diag_grads(mat): return sym_grad(mat)*jnp.eye(mat.shape[0])

        R_g = sym_diag_grads(R_g)
        Q_prior_g = vmap(vmap(sym_diag_grads))(Q_prior_g)
        Q_g = vmap(vmap(sym_diag_grads))(Q_g)

        # pack up
        lds_g = (b_prior_g, Q_prior_g, B_g, b_g, Q_g)
        hmm_g = (pi_g, A_g)
        gm_g = (R_g, lds_g, hmm_g)
        nn_g = (phi_g, theta_g)
        g = (gm_g, nn_g)

        # perform gradient updates
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return n_elbo, posteriors, params, opt_state


    @partial(jit, static_argnums=(4, 5))
    def infer_step(epoch_num, params, opt_state, x, # observations used in inference!!!!!!!!!!!!!!!!!!!!!
                   inference_iters, num_samples, burnin, key):
        """Perform inference without gradient step for eval purposes
               MLP parameters on the ELBO.
        """
        # unpack
        key, subkey = jrandom.split(key)
        R_est, lds_est, hmm_est = params[0]
        phi, theta = params[1]

        # always turn annealing off for eval
        nu = 1.

        # inference step
        n_elbo, posteriors = avg_neg_ELBO(x, R_est, lds_est, hmm_est, phi, # observations used to evaluate loss (ELBO)!!!!!!!!!!!!!!!!!!!!!
                                          theta, nu, subkey, inference_iters,
                                          num_samples)
        return n_elbo, posteriors


    # set plot
    fig, ax = plt.subplots(2, N, figsize=(10 * N, 6),
                           gridspec_kw={'height_ratios': [1, 2]})
    ax2 = ax
    for n in range(N):
        ax2[1, n] = ax[1, n].twinx()

    # train
    best_elbo = -jnp.inf
    for epoch in range(start_epoch, num_epochs):
        tic = time.time()
        niters = min(inference_iters, ((epoch // 100) + 1) * 5)
        key, trainkey = jrandom.split(key, 2)

        if args.eval_only:
            # infer posteriors for evaluation without grad step
            n_elbo, posteriors = infer_step(
                epoch, all_params, opt_state, x, niters, # observations used in inference!!!!!!!!!!!!!!!!!!!!!
                num_samples, burnin_len, trainkey)
        else:
            # training step
            n_elbo, posteriors, all_params, opt_state = training_step(
                epoch, all_params, opt_state, x, niters, # observations used in training!!!!!!!!!!!!!!!!!!!!!
                num_samples, burnin_len, trainkey)

        # evaluate
        qz, qzlag_z, qu, quu = posteriors
        mcc, _, sort_idx = matching_sources_corr(qz[0][:, :, 0], z_mu[:, :, 0])
        f_mu_est = vmap(decoder_mlp, in_axes=(None, -1),
                        out_axes=-1)(all_params[1][1], qz[0][:, :, 0])
        denoise_mcc = jnp.abs(jnp.diag(
            jnp.corrcoef(f_mu_est, f)[:M, M:])).mean()

        print("*Epoch: [{0}/{1}]\t"
              "ELBO: {2}\t"
              "mcc: {corr: .2f}\t"
              "denoise mcc: {dcorr: .2f}\t"
              "num. infernce iters: {3}\t"
              "eseed: {es}\t"
              "pseed: {ps}".format(epoch, num_epochs, -n_elbo, niters,
                                   corr=mcc, dcorr=denoise_mcc,
                                   es=args.est_seed, ps=args.param_seed))

        if epoch % plot_freq == 0 or args.eval_only:
            # plot
            plot_start = int(T/2)
            plot_len = 500
            plot_end = plot_start+plot_len
            for n in range(N):
                qz_mu_n = qz[0][sort_idx][n][plot_start:plot_end]
                qz_prec_n = qz[1][sort_idx][n][plot_start:plot_end]
                qu_n = jnp.exp(qu[sort_idx][n][plot_start:plot_end])
                u_n = states[n][plot_start:plot_end]
                z_mu_n = z_mu[n][plot_start:plot_end]
                plot_ic(u_n, z_mu_n, qu_n, qz_mu_n, qz_prec_n,
                        ax[0, n], ax[1, n], ax2[1, n])
            plt.pause(.5)

        # saving
        if -n_elbo > best_elbo:
            best_elbo = -n_elbo
            best_params = all_params
            best_posters = posteriors
            save_best(epoch, args, all_params, opt_state, tx)

        if args.eval_only:
            return best_params, best_posters, best_elbo
        print("Epoch took: ", time.time()-tic)
    return best_params, best_posters, best_elbo



