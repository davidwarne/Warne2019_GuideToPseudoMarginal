#=
# Generate trace plots, autocorrelations plots for a set of MCMC samplers
#
# print out R-hat and ESS for each parameter also
=#

using LaTeXStrings
using PyPlot 
using Random
using Distributions
using StatsBase
using KernelDensity
using MCMC
using JLD2

rc("font",family="serif")
rc("text",usetex="True")

# iteration index for burn-in samples
burn_start = 1
burn_stop = 8000

# iteration index for the tuned samples
sample_start = 1 + burn_stop
sample_stop = 15000 + burn_stop

d = 3
M = 4

lower = [0.0,0.0,0.0]
upper = [0.005,0.025,0.05]
θ_true = [0.001,0.005,0.01]

# load the chains
@load "michment.jdl" θ_t1 θ_t2 θ_t3 θ_t4

Σ_hat = cov(hcat(θ_t1[:,burn_start:burn_stop],
                      θ_t2[:,burn_start:burn_stop], 
                      θ_t3[:,burn_start:burn_stop], 
                      θ_t4[:,burn_start:burn_stop])')

# compute diagnostics for tuned chains
R,W,V,S = rnRhat([θ_t1[:,sample_start:sample_stop],
                  θ_t2[:,sample_start:sample_stop], 
                  θ_t3[:,sample_start:sample_stop], 
                  θ_t4[:,sample_start:sample_stop]])
println(R)
println(S)

# compute pooled mean and standard deviation for parameter estimates
μ = mean(hcat(θ_t1[:,sample_start:sample_stop],
              θ_t2[:,sample_start:sample_stop],
              θ_t3[:,sample_start:sample_stop],
              θ_t4[:,sample_start:sample_stop]),dims=2)

σ = std(hcat(θ_t1[:,sample_start:sample_stop],
             θ_t2[:,sample_start:sample_stop],
             θ_t3[:,sample_start:sample_stop],
             θ_t4[:,sample_start:sample_stop]),dims=2)

w = 7.143*2.0*0.394 # with inches
h = 5.001*3.0*0.394 # height inches

h1 = figure(1,figsize=(w/2,(h-0.7)/2))

# plot tuned chain behaviour
for i in 1:d
    # trace plot
    subplot(d,2,2*i-1)
    plot(θ_t1[i,sample_start:sample_stop],linewidth=0.75)
    plot([0,8000],[θ_true[i],θ_true[i]],"--r",linewidth=0.75)
    axis([0.0,8000,lower[i],upper[i]])
    ax = gca()
    ax.set_xticks([0.0,4000.0,8000.0])
    ax.set_yticks([lower[i],(lower[i]+upper[i])/2.0,upper[i]])
    xlabel(L"$m$")
    ylabel(latexstring(L"$k_","$i",L"$"))
 
    # autocorrelation function for lags 0-999
    subplot(d,2,2*i)
    ρ = autocor(θ_t1[i,sample_start:sample_stop],0:7999)
    plot(ρ,linewidth=0.75)
    axis([0.0,8000,-0.5,0.05])
    ax = gca()
    ax.set_xticks([0.0,4000.0,8000.0])
    ax.set_yticks([-0.5,0.0,0.5,1.0])
    xlabel(L"$\ell$")
    ylabel(latexstring(L"$\hat{\rho}_{","$i",L",\ell}$"))
end
h1.set_tight_layout(true)

h2 = figure(2,figsize=(w/2,(h-0.7)/2))
# plot tuned chain behaviour
for i in 1:d
    # trace plot
    subplot(d,2,2*i-1)
    plot(θ_t1[i,burn_start:burn_stop],linewidth=0.75)
    plot([0,8000],[θ_true[i],θ_true[i]],"--r",linewidth=0.75)
    axis([0.0,8000,lower[i],upper[i]])
    ax = gca()
    ax.set_xticks([0.0,4000.0,8000.0])
    ax.set_yticks([lower[i],(lower[i]+upper[i])/2.0,upper[i]])
    xlabel(L"$m$")
    ylabel(latexstring(L"$k_","$i",L"$"))
 
    # autocorrelation function
    subplot(d,2,2*i)
    ρ = autocor(θ_t1[i,burn_start:burn_stop],0:7999)
    plot(ρ,linewidth=0.75)
    axis([0.0,8000,-0.5,0.05])
    ax = gca()
    ax.set_xticks([0.0,4000.0,8000.0])
    ax.set_yticks([-0.5,0.0,0.5,1.0])
    xlabel(L"$\ell$")
    ylabel(latexstring(L"$\hat{\rho}_{","$i",L",\ell}$"))
end
h2.set_tight_layout(true)

w = 7.143*2.0*0.394 # with inches
h = 5.001*1.0*0.394 # height inches
h3 = figure(3,figsize=(w/2,(h-0.7)/2))
# plot marginal densities
for i in 1:d
    # create pooled sample
    s = cat(θ_t1[i,sample_start:sample_stop],
            θ_t2[i,sample_start:sample_stop],
            θ_t3[i,sample_start:sample_stop],
            θ_t4[i,sample_start:sample_stop],dims=1)

    # apply smoothed kernel density estimate
    kd = kde(s,bandwidth=0.075*(lower[i] + upper[i])/2.0)
    kx = range(lower[i],upper[i],length=1000) 
    # just the peak value rounded to nearest 10 for plotting
    mx = ceil(maximum(pdf(kd,kx))/10)*10

    subplot(1,d,i)
    plot(kx,pdf(kd,kx),linewidth=0.75)
    plot([θ_true[i],θ_true[i]],[0.0,mx],"--r",linewidth=0.75)
    axis([lower[i],upper[i],0.0,mx])
    ax = gca()
    ax.set_xticks([lower[i],(lower[i] + upper[i])/2.0,upper[i]])
    ax.set_yticks([0.0,mx/2.0,mx])
    xlabel(latexstring(L"$k_","$i",L"$"))
    ylabel(latexstring(L"$p(k_","$i",L" | \mathcal{D})$"))
end
h3.set_tight_layout(true)
