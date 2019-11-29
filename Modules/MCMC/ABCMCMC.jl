#=
# Functions for the Approximate Bayesian Computation MCMC Sampler.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

@doc """
    ABCMCMC(log_p,log_q,q,s,ρ,ϵ::Float64,D::Array{Float64},θ0::Array{Float64,1},burnin::Int,n::Int)

Generates `n` iterations of the ABC Markov Chain with stationary distribution 
p(θ | ρ(Ds,D) ≤ ϵ) where D is observed data, Ds ∼ s(D | θ) is the forward model,
and ρ(⋅,D) is an appropirately chosen discrepency metric to ensure that 
p(θ | ρ(Ds,D) ≤ ϵ) → p(θ | D) as ϵ → 0. Note the the 
proposal distribution q(θ* | θ) must sastisfy the usual conditions to ensure 
the Markov Chain is Ergodic.

Inputs:\n
    `log_p` - the log prior density function
    `log_q` - the log proposal density function
    `q` - proposal sampler
    `s` - forwards prblem simulation (i.e., generates simulated data given θ)
    `ρ` - the descrepancy metric
    `ϵ` - acceptance threshold
    `D` - observed data
    `θ0` - initial condition for the Markov Chain
    `burnin` - number of iterations to discard as burn-in samples
    `n` - number of iterations to to perfrom (after burn-in samples)


Outputs:\n
    `θ_t` - array of samples θt[:,i] is the Markov Chain state at the i-th 
            iteration. θ[j,:] is the trace of the j-th dimension of θt.
"""
function ABCMCMC(log_p,log_q,q,s,ρ,ϵ::Float64,D::Array{Float64},
                            θ0::Array{Float64,1},burnin::Int,n::Int)
    
    # Get dimenisionality of state space
    m, = size(θ0)

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
        # simulate forward model
        Ds = s(θ_p)
        
        if ρ(Ds,D) <= ϵ
            # compute acceptance probability (in log form)
            log_α = min(0.0, log_p(θ_p) + log_q(θ_t[:,j-1],θ_p) 
                         - log_p(θ_t[:,j-1]) - log_q(θ_p,θ_t[:,j-1]))
            # accept transition with prob α
            if log_u[i] <= log_α
                θ_t[:,j] = θ_p
            else # reject transition with prob 1 - α
                θ_t[:,j] = θ_t[:,j-1]
            end
        else # reject due to ρ(Ds,D) > ϵ
            θ_t[:,j] = θ_t[:,j-1]
        end
    end
    return θ_t
end
