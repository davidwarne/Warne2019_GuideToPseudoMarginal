#=
#
# Test SDE, runs example stochastic simulations using the Chemical Langevin
# SDE approximation and the Euler-Maruyama scheme.
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
using QuadGK

using SDE

rc("font",family="serif")
rc("text",usetex="True")

# initialise RNG
Random.seed!(51)

# Define propensity function vector and stoichiometric matrix for definition of 
# the Chemical Langevin equation (CLE) for the selected model
model,species,ν,a,θ,x0,T = SelectBCRN("mich-ment")

R = 1
X0 = x0.*ones(1,R)
T = 250.0
# simulate realisation using Euler-Maruyama scheme
Δt = 0.001
@time X,t = SimulateCLE(a,ν,θ,T,0.0,X0,Δt,R)

# desired plot dims
w = 3.75*2 # with inches
h = 2.625 # height inches
# plot realisations
h1 = figure(figsize=(w/2,(h-0.7)/2))
plot(t,X[1,:,:],linewidth=0.5);
plot(t,X[2,:,:],linewidth=0.5);
plot(t,X[3,:,:],"#edb120",linewidth=0.5);
plot(t,X[4,:,:],"#7f2f8e",linewidth=0.5);

xlabel(L"$t$")
ylabel(L"$\mathbf{X}_t$")
legend(species)
axis([0,250.0,0.0,100.])
ax = gca()
ax.set_yticks([0,25.0,50.0,75.0,100.0])
ax.set_xticks([0,50.0,100.0,150.0,200.0,250.0])
h1.set_tight_layout(true)

