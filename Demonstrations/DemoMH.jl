#=
# Test Metropolis-Hastings
#
# Simulates a number of Markov Chains targeting the Bayesian posterior, p(k1 | D),
# where D is observations from the stationary distribution of a 
# production-degradation model, k1 is the unknown production kinetic rate parameter
# and k2 = 0.01 is the known degradation rate.
#
# Plots are produced to demonstrate the effect of the proposal kernel variance 
# σ^2 on the chains convergence.
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

#= Uncomment this block to generate different data
Random.seed!(7)
model,species,ν,a,k,x0,T = SelectBCRN("prod-deg")
T = 100000.0
Δt = 0.01
R = 10
X0 = x0*ones(1,R)
# simulate realisations using Euler-Maruyama scheme
@time X_d,t_d = SimulateCLE(a,ν,k,T,0.0,X0,Δt,R,false)
=#

# data generated using seed=7
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

# Assume known parameter for degradation rate
k2 = 0.01

# log likelihood function
function log_like(D,θ,k2)
    # unnormalised log-likelihood of single observation
    g = (x) -> -2.0*x + (4.0*θ[1]/k2 - 1.0)*log(k2*x + θ[1])
    # unnormalised likelihood of single observation
    f = (x) -> exp(g(x)) 
    # compute normalising constant (catch Domain errors as non-convergent)
    Z,E = try
        quadgk(f,0,Inf)
    catch e 
        if isa(e,DomainError)
            (Inf,0.0)
        else
            throw(e)
        end
    end
    println(log(Z))
    # compute unnormalised log-likelihood of N independent observations
    ll = 0.0;
    for x in D
        ll += g(x) 
    end
    # normalise
    ll -= length(D)*log(Z)
    return ll
end

# prior k1 ~ U(0,2)
prior_dist = Uniform(0.0,2.0)
log_prior = (θ) -> logpdf(prior_dist,θ[1]) # log p(K), where p(k) = 1/(b-a)

# target log density (Bayesian posterior)
log_π = (θ) -> (isinf(log_prior(θ))) ? -Inf : log_prior(θ) + log_like(X_d,θ,k2) 

n = 10000
w = 7.143*2.0*0.394 # with inches
h = 5.001*3.0*0.394 # height inches
h1 = figure(1,figsize=(w/2,(h-0.7)/2))
w = 7.143*2.0*0.394 # with inches
h = 5.001*1.0*0.394 # height inches
h2 = figure(2,figsize=(w/2,(h-0.7)/2))
#evaluate exact posterior
f = (θ) -> exp(log_π(θ[1]))
Z,E = quadgk(f,-Inf,Inf)
xr = range(0.0,1.5,length=1000)
g = (θ) -> f(θ)/Z
yr = map(g,xr)
# Demonstrate effect of proposal kernel variance on acceptance rate and mixing
σ = [0.01,0.1,1.0]
for i in 1:length(σ)
    # Gaussian proposal pdf
    q_dist = Normal(0.0,σ[i])
    q_sim = (θ) -> rand(q_dist,1) + θ
    log_q_pdf = (θp,θ) -> logpdf(q_dist,θp[1]-θ[1])

    # try MCMC with Metropolis-Hastings
    @time θ_t = MetropolisHastings(log_π, log_q_pdf,q_sim,[0.8],0,n)


    figure(1)
    a_t = cumsum((θ_t[1,2:end] - θ_t[1,1:end-1]) .≠ 0.0)
    println(a_t[end])
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
   
    figure(2)
    plot(2:n,a_t,linewidth=0.75)
    axis([0.0,500.0,0.0,500.0])
    ax = gca()
    ax.set_xticks([0.0,5000.0,10000.0])
    ax.set_yticks([0.0,5000.0,10000.0])
    xlabel(L"$m$")
    ylabel(L"\mathcal{A}_m")

    figure(1)
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
ylabel(L"$p(k_1|\mathcal{D})$")
end
h1.set_tight_layout(true)
figure(2)
legend([L"\sigma = 0.01",L"\sigma= 0.1",L"\sigma = 1.0"])
h2.set_tight_layout(true)
