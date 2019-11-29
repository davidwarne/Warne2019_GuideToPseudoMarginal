#= Test Particle Markov chaim Monte Carlo
#
# TODO: have not selected the model of data to test yet. Dummy code for the moment.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

using PyPlot
using Random
using Distributions
using StatsBase
using LinearAlgebra
using QuadGK
using KernelDensity
using SDE
using MCMC

using JLD2

# initialise RNG
Random.seed!(513)

# Select the production-degradation model
model,species,ν,a,k,x0,T = SelectBCRN("schlogl")
 Δt = 0.01
σ_obs = 10.0
θ_true = copy(k)
g = (X) -> X + randn(1)*σ_obs
t_obs = [i for i=0.0:12.5:200.0]
 Y_obs = GenerateObservations(a,ν,θ_true,x0,Δt,t_obs[:],g)
 X_d = [t_obs';Y_obs]
 println(X_d)

# initialise RNG
Random.seed!(502)

# set up MCMC functions

# Bootstrap Particle Filter for Monte Carlo estimator to the log-likelihood 
function log_like_est(D,θ,N)
    Δt = 0.1
    X0 = x0*ones(1,N)

    # forwards evolution functions using Euler-Maruyama scheme
    function f(t,Xs,s)
        Xt, = SimulateCLE(a,ν,θ[:],t,s,Xs,Δt,N,false)
        return Xt[:,2,:]
    end
    
    # observation error model
    log_g = (Y,X) -> logpdf(Normal(0.0,σ_obs),Y[1] - X[1]) 

    t = D[1,:] # observation times
    Yt = D[2:end,:] # noisy observations
    # return result of Particle filter (first output)
    log_like, = BootstrapParticleFilter(log_g,f,X0,Yt,t)
    return log_like
end

# prior
lower = [0.0,0.0,0.0,0.0]
upper = [5.4e-1,7.5e-4,6.6e3,1.125e2]
prior_dist = Product(Uniform.(lower,upper))
log_prior = (θ) -> logpdf(prior_dist,θ) # log p(K), where p(k) = 1/(b-a)

# Bootstrap particle filter Monte Carlo estimator of target log density 
log_πhat = (θ,N) -> (isinf(log_prior(θ))) ? -Inf : log_prior(θ) + log_like_est(X_d,θ,N) 

N = 100

function initθ(M)
    θ0 = rand(prior_dist,M)
    i=1
    while i <= M
        ll = log_πhat(θ0[:,i],N)
        println(ll)
        if isinf(ll) == false
            i += 1
        else
            θ0[:,i] = rand(prior_dist,1)
        end
        println(i)
    end
    return θ0
end

θ0 = initθ(4)

@time println(log_πhat(θ0[:,1],N))
@time println(log_πhat(θ0[:,2],N))
@time println(log_πhat(θ0[:,3],N))
@time println(log_πhat(θ0[:,4],N))

#trial iterations using naive proposal
# proposal pdf
Σ = diagm(diag(cov(prior_dist)/100.0))
println(Σ)
q_dist = MvNormal(Σ)
q_sim = (θ) -> rand(q_dist) + θ
log_q_pdf = (θp,θ) -> logpdf(q_dist,θp-θ)

n = 20000
@time θ_t1 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θ0[:,1],N,0,n)
@time θ_t2 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θ0[:,2],N,0,n)
@time θ_t3 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θ0[:,3],N,0,n)
@time θ_t4 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θ0[:,4],N,0,n)
@save "schlogl.jdl" θ_t1 θ_t2 θ_t3 θ_t4

pooled_θ = hcat(θ_t1,θ_t2,θ_t3,θ_t4)
Σ_hat = cov(pooled_θ')
println(Σ_hat)
Σ_opt = (((2.38)^2)/4.0)*Σ_hat; 
q_dist = MvNormal(Σ_opt)
q_sim = (θ) -> rand(q_dist) + θ
log_q_pdf = (θp,θ) -> logpdf(q_dist,θp-θ)
n = 5000
for ii in 1:48
    # try MCMC with Pseudo-Marginal Metropolis-Hastings
    @load "schlogl.jdl" θ_t1 θ_t2 θ_t3 θ_t4
    # deepcopy 
    θpre_t1 = deepcopy(θ_t1)
    θpre_t2 = deepcopy(θ_t2)
    θpre_t3 = deepcopy(θ_t3)
    θpre_t4 = deepcopy(θ_t4)
    @time θ_t1 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θpre_t1[:,end],N,0,n)
    @time θ_t2 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θpre_t2[:,end],N,0,n)
    @time θ_t3 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θpre_t3[:,end],N,0,n)
    @time θ_t4 = PseudoMarginalMetropolisHastings(log_πhat, log_q_pdf,q_sim,θpre_t4[:,end],N,0,n)

    θ_t1 = hcat(θpre_t1,θ_t1)
    θ_t2 = hcat(θpre_t2,θ_t2)
    θ_t3 = hcat(θpre_t3,θ_t3)
    θ_t4 = hcat(θpre_t4,θ_t4)
    @save "schlogl.jdl" θ_t1 θ_t2 θ_t3 θ_t4
    R,W,V,S = rnRhat([θ_t1[:,20001:end],θ_t2[:,20001:end],θ_t3[:,20001:end],θ_t4[:,20001:end]])
    println(R)
    println(S)
end
@load "schlogl.jdl" θ_t1 θ_t2 θ_t3 θ_t4

h = figure()
subplot(221)
plot(θ_t1[1,:])
plot(θ_t2[1,:])
plot(θ_t3[1,:])
plot(θ_t4[1,:])
xlabel("1")
subplot(222)
plot(θ_t1[2,:])
plot(θ_t2[2,:])
plot(θ_t3[2,:])
plot(θ_t4[2,:])
xlabel("2")
subplot(223)
plot(θ_t1[3,:])
plot(θ_t2[3,:])
plot(θ_t3[3,:])
plot(θ_t4[3,:])
xlabel("3")
subplot(224)
plot(θ_t1[4,:])
plot(θ_t2[4,:])
plot(θ_t3[4,:])
plot(θ_t4[4,:])
xlabel("4")
