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
Random.seed!(513)

# define drift and diffusion coefficients for production-degradation Chemical 
# Langevin SDE
model,species,ν,a,k,x0,T = SelectBCRN("prod-deg")

# simulate realisation using Euler-Maruyama scheme
R = 4;
T = 500.0; # overwriting default T
Δt = 0.1;
X0 = x0*ones(1,R)
@time X,t = SimulateCLE(a,ν,k,T,0.0,X0,Δt,R)

# simulate single long running realisation using Euler-Maruyama scheme
R = 1;
T_inf = 1000000.0; # large enought to reach stationarity
Δt = 0.1; 
X0 = x0*ones(1,R)
@time X_inf,t_inf = SimulateCLE(a,ν,k,T_inf,0.0,X0,Δt,R)

# desired plot dims
w = 7.143*0.394 # with inches
h = 5.001*0.394 # height inches
# plot realisations
h1 = figure(figsize=(w/2,(h-0.7)/2))
for i in 1:length(x0)
    plot(t,X[i,:,:],"b",linewidth=0.5,alpha=0.5);
end
xlabel(L"$t$")
ylabel(L"$X_t$")
axis([0.0, T,0.0, 150.0])
h1.set_tight_layout(true)
ax = gca()
ax.set_xticks([0.0,125.0,250.0,375.0,500.0])
# plot stationary distribution (simulation vs Analytic)
h2 = figure(figsize=(w/2,(h-0.7)/2))

# using simulation
n,bins, patches = plt.hist(X_inf[1,:,:], 40, density=1,edgecolor="black",linewidth=0.5,orientation="horizontal",alpha=0.5)

# using Analytically derived Stationary distribution
ps = (x) -> exp(-2.0*x + (4.0*k[1]/k[2] - 1.0)*log(k[2]*x + k[1]))
# compute normalising constant
Z,E = quadgk(ps,0,Inf)
y = zeros(1000)
x = collect(range(0.0,150.0,length=1000))
@. y = ps(x)/Z 

# plot analytic solution
plot(y, x,"r--",linewidth=0.5)
axis([0.0,0.04,0.0,150.0])
ylabel(L"$X_\infty$")
xlabel(L"$p_s(X_\infty)$")
legend(["Exact","Simulations"],loc="lower right")
h2.set_tight_layout(true)
