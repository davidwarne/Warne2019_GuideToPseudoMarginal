#=
# Test Approximate Bayesian Computation Metropolis-Hastings
#
# Simulates a number of Markov Chains targeting the approximate Bayesian posterior, 
# p(k1 | ρ(Ds,D) ≤ ϵ), where D is observations from the stationary distribution of a 
# production-degradation model,Ds is simulated data, ρ(⋅,D) is a discrepancy 
# metric, ϵ is the acceptance threshold,  k1 is the unknown production kinetic 
# rate parameter and k2 = 0.01 is the known degradation kinetic rate parameter. 
#
# Plots are produced to demonstrate the effect of the acceptance threshold.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

using LaTeXStrings
using PyPlot
using Random
using Distributions
using KernelDensity

using QuadGK

using SDE
using MCMC

rc("font",family="serif")
rc("text",usetex="True")

# data generated as per testMH.jl
X_d = [91.67837087481476,
 101.63712230369346,
  88.12935759760977,
  98.88364052988958,
  96.3607300094281 ,
 119.58953067522995,
 100.61686408828994,
 105.10877557845691,
 105.30281066055595,
  96.99987592534579]

# initialise RNG
Random.seed!(502)

# set up MCMC functions

# know parameter for degradation rate
k2 = 0.01

# Select the production-degradation model
model,species,ν,a,k,x0,T = SelectBCRN("prod-deg")

# model simulation process 
function sim(θ)
    if θ[1] < 0.0
        return zeros(size(X_d))
    end
    R = length(X_d)
    k = [θ[1], k2]
    T = 1000.0
    Δt = 1.0
    X0 = x0*ones(1,R)
    # simulate realisations using Euler-Maruyama scheme
    X_s,t_s = SimulateCLE(a,ν,k,T,0.0,X0,Δt,R,false)
    return X_s[1,2,:]
end

# Descrepancy metric
function ρ(Ds,D)
    return abs(mean(D) - mean(Ds)) + abs(std(D) - std(Ds))
end

# prior
prior_dist = Uniform(0.0,2.0)
log_prior = (θ) -> logpdf(prior_dist,θ[1]) # log p(K), where p(k) = 1/(b-a)

# proposal pdf
q_dist = Normal(0.0,0.1)
q_sim = (θ) -> rand(q_dist,1) + θ
log_q_pdf = (θp,θ) -> logpdf(q_dist,θp[1]-θ[1])

n = 10000
# log likelihood
function log_like(D,θ,k2)
    # compute normalising constant
    g = (x) -> -2.0*x + (4.0*θ[1]/k2 - 1.0)*log(k2*x + θ[1])
    f = (x) -> exp(g(x)) 
    Z,E = try
        quadgk(f,0,Inf)
    catch e 
        if isa(e,DomainError)
            (Inf,0.0)
        else
            throw(e)
        end
    end
    ll = 0.0;
    for x in D
        ll += g(x) 
    end
    ll -= length(D)*log(Z)
    return ll
end

# target log density (Bayesian posterior)
log_π = (θ) -> (isinf(log_prior(θ))) ? -Inf : log_prior(θ) + log_like(X_d,θ,k2) 
f = (θ) -> exp(log_π(θ[1]))
Z,E = quadgk(f,-Inf,Inf)
xr = range(0.0,2.0,length=1000)
g = (θ) -> f(θ)/Z
yr = map(g,xr)


# demonstrate effect of ϵ on stationary distribution
w = 7.143*2.0*0.394 # with inches
h = 5.001*3.0*0.394 # height inches
h = figure(figsize=(w/2,(h-0.7)/2))
ϵ = [14.0,7.0,3.5]
for i in 1:length(ϵ)

    # try MCMC with Pseudo-Marginal Metropolis-Hastings
    @time θ_t = ABCMCMC(log_prior,log_q_pdf,q_sim,sim,ρ,ϵ[i],X_d,[0.8],0,n)

    # plot trace and stationary distribution estimate
    subplot(3,2,2*i-1)
    plot(1:n,θ_t',linewidth=0.75); 
    plot([0.0,500.0],[1.0,1.0],"--r",linewidth=0.75)
    axis([0.0,500.0,0.8,1.2])
    ax = gca()
    ax.set_xticks([0.0,250.0,500.0])
    ax.set_yticks([0.8,1.0, 1.2])
    xlabel(L"$m$")
    ylabel(L"$k_1$")
    subplot(3,2,2*i)
    for j in (250,500,10000)
        kd = kde(θ_t[1,1:j])
        kx = range(0.0,1.5,length=1000)
        plot(kx,pdf(kd,kx),linewidth=0.75)
    end
    plot(xr,yr,"--k",linewidth=0.75)
    plot([1.0,1.0],[0.0,20.0],"--r",linewidth=0.75)
    axis([0.8,1.2,0.0,20.0])
    ax = gca()
    ax.set_xticks([0.8,1.0,1.2])
    ax.set_yticks([0.0,10.0,20.0])
    xlabel(L"$k_1$")
    ylabel(L"$p(k_1|\rho(\mathcal{D},\mathcal{D}_s) \leq \epsilon)$")
end
h.set_tight_layout(true)
