#=
# Particle filter for evaluation of the log-likelihood of a partially observed
# solution to a stochastic differential equation. Given true state x_t, a observation 
#  is distributed according to Y_t ~ g(y_t | x_t), the transition probabilities 
#  for the stochastic process {X_t, t ≥ 0} is given by the solution to the forwards
#  Kolmogorov equation X_t ∼ f(x_t|x_{t-1}) which may be sampled using stochastic
#  simulation.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

@doc """
    BootstrapParticleFilter(log_g,f,X0::Array{Float64,2},Yt::Array{Float64,2},t)

Generates weighted samples from the distributions π(X_t | Y_1:t-1) for t = 2,...,T
and produces unbiased estimates of the marginal likelihoods π(Y_t | Y_1:t-1) and
the full joint likelihood π(Y_1:T).

Inputs:\n
    `log_g` - log probability density of the observation noise model 
              i.e., Y_t ∼ g(Y_t | X_t)
    `f` - Simultion process for the state-space model. Given M samples {X_t^i}_{i=1}^M,
          generates X_{t+1}^i ∼ f(X_{t+1} | X_t^i) for i = 1,2,...,M
    `X0` - N×M array of samples from initial distribution of true states assumed 
           that X0 ∼ π(X_0 | Y_0).
    `Yt` - time series of observations
    `t` - observation times (assumed exact)

Outputs:\n
    `log_like` - full joint log likelihood log π(Y_1:T)
    `log_like_t` - T×1 array of marginal log likelihoods log π(Y_t | Y_1:t-1)
    `Xt` - N×M×T array of samples with X[:,:,j] is the collection of samples from 
            π(X_j | Y_1:j)
    `log_wt` - M×T array of unnormalised log weights, exp(log_wt[i,j] is the 
                unnormalised weight of sample Xt[:,i,j]
"""
function BootstrapParticleFilter(log_g,f,X0::Array{Float64,2},Yt::Array{Float64,2},
                                 t::Array{Float64,1})

    # determine number of Particles and their dimension
    N,M = size(X0)
    # determine the number of timesteps
    T = length(t)

    # initialise memory for particles, weights and marginal log likelihoods
    Xt = zeros(N,M,T)
    log_wt = zeros(M,T)
    wtn = zeros(M)
    wt = zeros(M)
    log_like_t = -Inf*ones(T)

    # first iteration is just the initial distribution, implicitly Y0 = X0
    # i.e., perfect initial condition observation
    Xt[:,:,1] = X0
    log_wt[:,1] .= -log(M)  # (X0,w0) ≈ π(X0|Y0)
    log_like_t[1] = 0.0

    # start particle filter
    for j in 2:T
        # propagate particles forward X_t ∼ f(X_t | X_{t-1})
        Xt[:,:,j] = f(t[j],Xt[:,:,j-1],t[j-1])

        # set unnormalised log weights
        for i in 1:M
            log_wt[i,j] = log_g(Yt[:,j],Xt[:,i,j])
        end
        # compute marginal log-likelihood
        log_wt_max = maximum(log_wt)
        @. wt = exp(log_wt[:,j] - log_wt_max)
        log_like_t[j] = log(sum(wt)) + log_wt_max - log(M)

        if isinf(log_like_t[j]) # break on vanishing marginal likelihood
            break; 
        end
        # normalise weights and perform importance resampling
        # to obtain (Xt,wt) ≈ π(Xt | Y_1:t)
        @. wtn = exp(log_wt[:,j] - log_like_t[j])/M
        I = sample(collect(1:M),Weights(wtn),M,replace=true)
        Xt[:,1:M,j] = Xt[:,I,j]        
    end
    # compute joint log likelihood
    log_like = sum(log_like_t)
    return log_like, log_like_t, Xt, log_wt
end
