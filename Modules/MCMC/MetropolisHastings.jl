#=
# Functions for the exact Metropolis-Hastings Sampler.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

@doc """
    MetropolisHastings(log_π,log_q,q,θ0::Array{Float64,1},burnin::Int,n::Int)

Generates `n` iterations of the a Markov Chain with stationary distribution 
π(θ) using the Metropolis-Hastings MCMC algorithm. Note the the proposal distribution
q(θ* | θ) must sastisfy the usual conditions to ensure the Markov Chain is 
π-Ergodic.

Inputs:\n
    `log_π` - natural logarithm of target propability density function ln π(θ). 
              π(θ) need only be known up to a constant of proportionality.
    `log_q` - natural logarithm of proposal density ln q(θ*|θ)
    `q` - proposal density sampler
    `θ0` - initial condition for the Markov Chain
    `burnin` - number of iterations to discard as burn-in samples
    `n` - number of iterations to to perfrom (after burn-in samples)


Outputs:\n
    `θ_t` - array of samples θt[:,i] is the Markov Chain state at the i-th 
            iteration. θ[j,:] is the trace of the j-th dimension of θt.
"""
function MetropolisHastings(log_π,log_q,q,θ0::Array{Float64,1},burnin::Int,n::Int)
    
    # Get dimenisionality of state space
    m, = size(θ0)

    # initialise previous log π(θ) to avoid repeat evaluation
    prev_log_π = log_π(θ0)
    cur_log_π = prev_log_π

    # allocate memory for θ_t
    θ_t = zeros(Float64,m,n)
    # initialise θ_t
    θ_t[:,1] = θ0

    # generate array of u ~ U(0,1) for 
    log_u = rand(burnin+n) # sample uniform
    @. log_u = log(log_u)  # broadcast log 

    j = 2
    # perform Metropolis-Hastings interations
    for i in 2:burnin+n

        if i <= burnin 
            #  burn-in period, set prev to j=1 and cur to j=2
            j = 2
            θ_t[:,j-1] = θ_t[:,j]
        else # offset MCMC index
            j = i - burnin
        end
    
        # generate proposal θ_p ~ q(⋅ | θ_j)
        θ_p = q(θ_t[:,j-1])
        
        # compute log π(θ_p)
        cur_log_π = log_π(θ_p)
        
        # compute acceptance probability (in log form)
        log_α = min(0.0, cur_log_π + log_q(θ_t[:,j-1],θ_p) 
                         - prev_log_π - log_q(θ_p,θ_t[:,j-1]))
        # accept transition with prob α
        if log_u[i] <= log_α
            θ_t[:,j] = θ_p
            # store for next iteration
            prev_log_π = cur_log_π
        else # reject transition with prob 1 - α
            θ_t[:,j] = θ_t[:,j-1]
        end
    end
    return θ_t
end
