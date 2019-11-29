"""
A Julia module to demonstrate various MCMC schemes, both exact and approximate
sampling is supported.

Supported Samplers:
    Metropolis-Hastings MCMC, Approximate Bayesian Computation MCMC, 
    Pseudo-Marginal (Particle) MCMC 
"""
module MCMC
#=
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#
using StatsBase
using Distributions
export MetropolisHastings
export PseudoMarginalMetropolisHastings
export ABCMCMC
export BootstrapParticleFilter

export splitRhat
export rnRhat

include("MetropolisHastings.jl")
include("ABCMCMC.jl")
include("PseudoMarginalMetropolisHastings.jl")
include("BootstrapParticleFilter.jl")
include("Diagnostics.jl")

end
