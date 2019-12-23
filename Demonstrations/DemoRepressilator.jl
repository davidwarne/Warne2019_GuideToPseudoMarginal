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
model,species,ν,a,θ,x0,T = SelectBCRN("repressilator")

R = 1
X0 = x0.*ones(1,R)
T = 100.0
# simulate realisation using Euler-Maruyama scheme
Δt = 0.001
@time X,t = SimulateCLE(a,ν,θ,T,0.0,X0,Δt,R)

# desired plot dims
w = 3.75*2.0 # with inches
h = 2.625*1.5 # height inches
# plot realisation
h1 = figure(figsize=(w/2,(h-0.7)/2))
for i in 1:length(x0)
plot(t,X[i,:,:],linewidth=0.5);
end
xlabel(L"$t$")
ylabel(L"$\mathbf{X}_t$")
legend(species)
axis([0.0,100.0,0.0,1000.0])
ax = gca()
ax.set_xticks([0,25.0,50.0,75.0,100.0])
ax.set_yticks([0,250.0,500.0,750.0,1000.0])
h1.set_tight_layout(true)

