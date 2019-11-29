#= Plot Particle Filter example
#
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
using StatsBase
using LinearAlgebra
using QuadGK
using KernelDensity
using SDE
using MCMC

using JLD2

rc("font",family="serif")
rc("text",usetex="True")
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
X0 = Y_obs[5]*ones(1,4)

ti = t_obs[5:8]
Yt = Y_obs[5:8]

# determine number of Particles and their dimension
N,M = size(X0)

# determine the number of timesteps
T = length(ti)

# initialise memory for particles, weights and marginal log likelihoods
Xt = zeros(N,M,T)
log_wt = zeros(M,T)
wtn = (1/M)*ones(M)
wt = zeros(M)
log_like_t = -Inf*ones(T)
# first iteration is just the initial distribution, implicitly Y0 = X0
# i.e., perfect initial condition observation
Xt[:,:,1] = X0
log_wt[:,1] .= -log(M)  # (X0,w0) ≈ π(X0|Y0)
log_like_t[1] = 0.0
Ic = 1:M

w = 7.143*2.0*0.394 # with inches
h = 5.001*3.0*0.394 # height inches
h1 = figure(figsize=(w/2,(h-0.7)/2))
log_g = (Y,X) -> logpdf(Normal(0.0,σ_obs),Y[1] - X[1]) 
# start particle filter
j = 2
# propagate particles forward X_t ∼ f(X_t | X_{t-1})
println(j) 
println(ti[j-1]) 
println(ti[j]) 
X,t = SimulateCLE(a,ν,θ_true,ti[j],ti[j-1],Xt[:,:,j-1],Δt,M,true)
Xt[:,:,j] = X[:,end,:]

subplot(4,2,2)
bar(X[1,1,1:4]+[-54.0,-27.0, 0.0, 27.0],[0.25,0.25,0.25,0.25],width=25.0,color="#aaaaaa",align="center",alpha=0.75)
ax = gca()
ax.set_yticks([0.0,0.5,1.0])
ax.set_xticks([0,200.0,400.0,600.0])
for i in 2:4
    subplot(4,2,2*i - 1)
    plot(t,X[1,:,1],"-",color="#1b9e77",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,2],"-",color="#1b9e77",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,3],"-",color="#1b9e77",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,4],"-",color="#1b9e77",linewidth=1.0,alpha=0.5)
end

# set unnormalised log weights
for i in 1:M
    log_wt[i,j] = log_g(Yt[j],Xt[:,i,j])
end

# compute marginal log-likelihood
log_wt_max = maximum(log_wt)
@. wt = exp(log_wt[:,j] - log_wt_max)
log_like_t[j] = log(sum(wt)) + log_wt_max - log(M)

# normalise weights and perform importance resampling
# to obtain (Xt,wt) ≈ π(Xt | Y_1:t)
@. wtn = exp(log_wt[:,j] - log_like_t[j])/M
println(wtn)
I = sample(collect(1:M),Weights(wtn),M,replace=true)
Xt[:,1:M,j] = Xt[:,I,j]        
print(I)

subplot(4,2,4)
bar(X[1,end,4],[0.82],width=25.0,color="#1b9e77",align="center",alpha=0.75)
bar(X[1,end,1:3],[0.04,0.09,0.05],width=25.0,color="#1b9e77",align="center",alpha=0.5)
ax = gca()
ax.set_yticks([0.0,0.5,1.0])
ax.set_xticks([0,200.0,400.0,600.0])

j = 3
# propagate particles forward X_t ∼ f(X_t | X_{t-1})
println(j) 
println(ti[j-1]) 
println(ti[j]) 
X,t = SimulateCLE(a,ν,θ_true,ti[j],ti[j-1],Xt[:,:,j-1],Δt,M,true)
Xt[:,:,j] = X[:,end,:]

for i in 3:4
    subplot(4,2,2*i - 1)
    plot(t,X[1,:,4],"-",color="#d95f02",linewidth=1.0,alpha=0.5)
    plot(t,X[1,:,1],"-",color="#d95f02",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,2],"-",color="#d95f02",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,3],"-",color="#d95f02",linewidth=0.25,alpha=0.25)
end

# set unnormalised log weights
for i in 1:M
    log_wt[i,j] = log_g(Yt[j],Xt[:,i,j])
end

# compute marginal log-likelihood
log_wt_max = maximum(log_wt)
@. wt = exp(log_wt[:,j] - log_wt_max)
log_like_t[j] = log(sum(wt)) + log_wt_max - log(M)

# normalise weights and perform importance resampling
# to obtain (Xt,wt) ≈ π(Xt | Y_1:t)
@. wtn = exp(log_wt[:,j] - log_like_t[j])/M
println(wtn)
I = sample(collect(1:M),Weights(wtn),M,replace=true)
Xt[:,1:M,j] = Xt[:,I,j]        
print(I)
subplot(4,2,6)
bar(X[1,end,4],[0.72],width=25.0,color="#d95f02",align="center",alpha=0.75)
bar(X[1,end,1:3],[0.14,0.04,0.1],width=25.0,color="#d95f02",align="center",alpha=0.5)
ax = gca()
ax.set_yticks([0.0,0.5,1.0])
ax.set_xticks([0,200.0,400.0,600.0])

j = 4
# propagate particles forward X_t ∼ f(X_t | X_{t-1})
println(j) 
println(ti[j-1]) 
println(ti[j]) 
X,t = SimulateCLE(a,ν,θ_true,ti[j],ti[j-1],Xt[:,:,j-1],Δt,M,true)
Xt[:,:,j] = X[:,end,:]

for i in 4:4
    subplot(4,2,2*i - 1)
    plot(t,X[1,:,4],"-",color="#7570b3",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,1],"-",color="#7570b3",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,2],"-",color="#7570b3",linewidth=0.25,alpha=0.25)
    plot(t,X[1,:,3],"-",color="#7570b3",linewidth=1.0,alpha=0.5)
end

# set unnormalised log weights
for i in 1:M
    log_wt[i,j] = log_g(Yt[j],Xt[:,i,j])
end

# compute marginal log-likelihood
log_wt_max = maximum(log_wt)
@. wt = exp(log_wt[:,j] - log_wt_max)
log_like_t[j] = log(sum(wt)) + log_wt_max - log(M)

# normalise weights and perform importance resampling
# to obtain (Xt,wt) ≈ π(Xt | Y_1:t)
@. wtn = exp(log_wt[:,j] - log_like_t[j])/M
println(wtn)
I = sample(collect(1:M),Weights(wtn),M,replace=true)
Xt[:,1:M,j] = Xt[:,I,j]        
print(I)

subplot(4,2,8)
bar(X[1,end,3],[0.61],width=25.0,color="#7570b3",align="center",alpha=0.75)
bar(X[1,end,1:3],[0.23,0.06,0.1],width=25.0,color="#7570b3",align="center",alpha=0.5)
ax = gca()
ax.set_yticks([0.0,0.5,1.0])
ax.set_xticks([0,200.0,400.0,600.0])
for i in 1:4
    subplot(4,2,2*i - 1)
    uerr=[30.0 30.0 30.0 30.0]
    lerr=[30.0 30.0 30.0 30.0]
    errorbar(ti,Yt',yerr=[lerr;uerr],fmt="ok",ms=2.0,capsize=3.0);
    ax = gca()
    ax.set_xticks(ti)
    ax.set_yticks([0,200.0,400.0,600.0])
    ylabel(L"$X_t$")
    xlabel(L"$t$")
    subplot(4,2,2*i)
    ax = gca()
    ax.set_yticks([0.0,0.5,1.0])
    ax.set_xticks([0,200.0,400.0,600.0])
    ylabel(latexstring(L"$W_","$(i-1)",L"^{(k)}$"))
    xlabel(latexstring(L"$X_","$(i-1)",L"^{(k)}$"))
end
h1.set_tight_layout(true)
