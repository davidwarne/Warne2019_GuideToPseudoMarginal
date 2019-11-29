#=
# Euler-Maruyama method for numerical integration of Chemical Langevin SDEs 
# of Ito form. These functions allow the user to prescribe the stoichiometric matrix 
# and a vector valued propensity function, and the correct drift and diffusion 
# functions are automatically constructed. This is a more natural for defining a
# chemical network versus the more generic Euler-Maruyama function.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Queensland University of Technology
=#

@doc """
    SimulateCLE(a::Function,ν::Array{Float64,2},k::Array{Float64,1},T::Float64,t0::Float64, X0::Array{Float64,2},n::Int,R::Int,paths=true) 

Simulates, using the Euler-Maruyama method, `R` realisations of the Chemical 
Langevin SDE approximation to a stochastic chemical reaction network of `N` 
chemical species and `M` chemical reactions. The SDE has the form

        dX_t = ν⋅a(X_t;k)dt + ν⋅diag(a(X_t;k)^(1/2))dW_t, 

where `{X_t,t ≥ 0}` is an N-dimensional continuous state Markov process,
`{W_t, t ≥ 0}` is an M-dimensional Wiener process (one per reaction channel), 
`ν ∈ R^(N×M)` is the stoichiometric matrix, and `a : R^N -> R^N` is the vector 
propensity function. `a` is parameterised by the vector `k ∈ R^d`. Usually `k` 
is a vector of kinetic rate parameters for each reaction, in such a case `d = M`. 
However, this is not required in general.

Inputs:\n
    `a` - propensity function
    `ν` - stoichiometric matrix 
    `k` - parameter vector
    `T` - final time
    `t0` - initial time
    `X0` - Initial condition, must be an N×R matrix (one condition per realisation)
    `n` - number of timesteps
    `R` - number of realisations
    `paths` - true (default) to output full sample paths or false to return only X_T
Outputs: \n
    `X` - an N×n×R matrix, with `X[:,:,j]` is the `j`th realisation and `X[:,i,:]` is the 
         value of `X` at the `i`th timestep for all realisations.
    `t` - a vector of time points. `X[:,i,:]` is the solution at time `t[i]`. 
"""
function SimulateCLE(a::Function,ν::Array{Float64,2},k::Array{Float64,1},T::Float64,
                t0::Float64,X0::Array{Float64,2},n::Int,R::Int,paths=true) 
    
    # generic form of the CLE drift and diffusion
    α = (X,t,k) -> ν*a(X,k)
    σ = (X,t,k) -> ν*diagm(a(X,k).^(1/2))
    
    # simulate realisations using Euler-Maruyama scheme
    X,t = Euler_Maruyama(α,σ,k,T,t0,X0,Δt,R,paths)
    #t .+= t0
    return X,t
end

@doc """
    SimulateCLE(a::Function,ν::Array{Float64,2},k::Array{Float64,1},T::Float64,X0::Array{Float64,2},n::Int,R::Int) 

Simulates, using the Euler-Maruyama method, `R` realisations of the Chemical 
Langevin SDE approximation to a stochastic chemical reaction network of `N` 
chemical species and `M` chemical reactions. The SDE has the form

        dX_t = ν⋅a(X_t;k)dt + ν⋅diag(a(X_t;k)^(1/2))dW_t, 

where `{X_t,t ≥ 0}` is an N-dimensional continuous state Markov process,
`{W_t, t ≥ 0}` is an M-dimensional Wiener process (one per reaction channel), 
`ν ∈ R^(N×M)` is the stoichiometric matrix, and `a : R^N -> R^N` is the vector 
propensity function. `a` is parameterised by the vector `k ∈ R^d`. Usually `k` 
is a vector of kinetic rate parameters for each reaction, in such a case `d = M`. 
However, this is not required in general.

Inputs:\n
    `a` - propensity function
    `ν` - stoichiometric matrix 
    `k` - parameter vector
    `T` - final time
    `t0` - initial time
    `X0` - Initial condition, must be an N×R matrix (one condition per realisation)
    `Δt` - timestep
    `R` - number of realisations
    `paths` - true (default) to output full sample paths or false to return only X_T
Outputs: \n
    `X` - an N×n×R matrix, with `X[:,:,j]` is the `j`th realisation and `X[:,i,:]` is the 
         value of `X` at the `i`th timestep for all realisations.
    `t` - a vector of time points. `X[:,i,:]` is the solution at time `t[i]`. 
"""
function SimulateCLE(a::Function,ν::Array{Float64,2},k::Array{Float64,1},T::Float64,
                t0::Float64,X0::Array{Float64,2},Δt::Float64,R::Int,paths=true) 
    
    # generic form of the CLE drift and diffusion
    α = (X,t,k) -> ν*a(X,k)
    σ = (X,t,k) -> ν*diagm(a(X,k).^(1/2))
    
    # simulate realisations using Euler-Maruyama scheme
    X,t = Euler_Maruyama(α,σ,k,T,t0,X0,Δt,R,paths)
    #t .+= t0
    return X,t
end
