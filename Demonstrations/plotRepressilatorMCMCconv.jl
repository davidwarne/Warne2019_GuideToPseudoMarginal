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
burn_stop = 5000

# iteration index for the tuned samples
sample_start = 1 + burn_stop
sample_stop = 135000 + burn_stop

d = 4
M = 4

lower = [500.0,0.0,0.0,0.0]
upper = [2500.0,10.0,20.0,10.0]
θ_true = [1000.0,1.0,5.0,2.0]
θ_labels = [raw"\alpha",raw"\alpha_0",raw"\beta",raw"n"]

# load the chains
@load "repressilator.jdl" θ_t1 θ_t2 θ_t3 θ_t4

Σ_hat = cov(hcat(θ_t1[:,burn_start:burn_stop],
                      θ_t2[:,burn_start:burn_stop], 
                      θ_t3[:,burn_start:burn_stop], 
                      θ_t4[:,burn_start:burn_stop])')
println(Σ_hat)
# compute diagnostics for initialisation chains
Rb,Wb,Vb,Sb = rnRhat([θ_t1[:,burn_start:burn_stop],
                      θ_t2[:,burn_start:burn_stop], 
                      θ_t3[:,burn_start:burn_stop], 
                      θ_t4[:,burn_start:burn_stop]])
println(Rb)
println(Sb)

# compute diagnostics for tuned chains
R,W,V,S = rnRhat([θ_t1[:,sample_start:sample_stop],
                  θ_t2[:,sample_start:sample_stop], 
                  θ_t3[:,sample_start:sample_stop], 
                  θ_t4[:,sample_start:sample_stop]])
println(R)
println(S)

# compute pooled mean and standard deviation
μ = mean(hcat(θ_t1[:,sample_start:sample_stop],
              θ_t2[:,sample_start:sample_stop],
              θ_t3[:,sample_start:sample_stop],
              θ_t4[:,sample_start:sample_stop]),dims=2)
σ = std(hcat(θ_t1[:,sample_start:sample_stop],
             θ_t2[:,sample_start:sample_stop],
             θ_t3[:,sample_start:sample_stop],
             θ_t4[:,sample_start:sample_stop]),dims=2)

w = 7.143*2.0*0.394 # with inches
h = 5.001*2.0*0.394 # height inches
h3 = figure(3,figsize=(w/2,(h-0.7)/2))

# plot marginal densities
for i in 1:d
    # create pooled sample
    #s = θ_t4[i,sample_start:sample_stop]
    s = cat(θ_t1[i,sample_start:sample_stop],
            θ_t2[i,sample_start:sample_stop],
            θ_t3[i,sample_start:sample_stop],
            θ_t4[i,sample_start:sample_stop],dims=1)

    # apply smoothed kernel density estimate
    kd = kde(s,bandwidth=0.05*(lower[i] + upper[i])/2.0)
    kx = range(lower[i],upper[i],length=1000) 
    # just the peak value rounded to nearest 10 for plotting
    mx = maximum(pdf(kd,kx))
    println(mx)

    mx  = (mx >= 10) ? ceil(mx/10)*10 : mx

    subplot(d/2,d/2,i)
    ax = gca()
    ax.ticklabel_format(style="scientific",axis="both",scilimits=(-2,4))
    plot(kx,pdf(kd,kx),linewidth=0.75)
    plot([θ_true[i],θ_true[i]],[0.0,mx],"--r",linewidth=0.75)
    axis([lower[i],upper[i],0.0,mx])
    ax.set_xticks([lower[i],(lower[i] + upper[i])/2.0,upper[i]])
    ax.set_yticks([0.0,mx/2.0,mx])

    xlabel(latexstring(L"$",θ_labels[i],L"$"))
    ylabel(latexstring(L"$p(",θ_labels[i],L" | \mathcal{D})$"))
end
h3.set_tight_layout(true)
