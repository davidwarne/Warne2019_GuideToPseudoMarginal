#=
# Euler-Maruyama method for numerical integration of scalar SDEs in Ito form  
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Queensland University of Technology
=#

@doc """
    Euler_Maruyama(α::Function, σ::Function, θ::Array{Float64,1}, T::Float64, t0::Float64, X0::Array{Float64}, n::Int, R::Int,paths=true) 

Simulates `R` realisations of a non-autonomous vector stochastic differential 
equation of the form

        dX_t = α(X_t,t;θ)dt + σ(X_t,t;θ)dW_t, 

where `{X_t,t ≥ 0}` is an N-dimensional continuous state Markov process,
`{W_t, t ≥ 0}` is an M-dimensional Wiener process, `α(⋅,⋅;θ) : R^N × T → R^N` 
is the drift function, and `σ(⋅,⋅; θ) R^N × T → R^(N×M)` is the diffusion function.
Both `α` and `σ` are parameterised by the vector `θ ∈ R^d`.

Inputs:\n
    `α` - drift function
    `σ` - diffusion function 
    `θ` - parameter vector
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
    `Δt` - timestep used in simulation
"""
function Euler_Maruyama(α::Function, σ::Function, θ::Array{Float64,1}, 
                        T::Float64,t0::Float64, X0::Array{Float64},n::Int,
                        R::Int,paths=true)
    # Get dimensions
    N,M = size(σ(X0[:,1],0,θ)) # σ ∈ R^(N×M)
    
    # time step
    Δt = (T - t0)/n
    t = [(i-1)*Δt for i=1:n+1] .+ t0
    if paths
        # initialise output matrix 
        X = zeros(N,n+1,R) 
        # evolve the realisations using Euler Maruyama steps
        for j in 1:R
            # Generate Brownian increments for this realisation
            ΔW = sqrt(Δt)*randn(M,n)
            #initialise this realisation and simulate
            X[:,1,j] = X0[:,j]
            for i in 1:n
                X[:,i+1,j] = X[:,i,j] + α(X[:,i,j],t[i],θ)*Δt + σ(X[:,i,j],t[i],θ)*ΔW[:,i]
            end
        end
        # return full paths
        return X,t,Δt
    else
        # initialise output matrix 
        X = zeros(N,2,R)
        # evolve the realisations using Euler Maruyama steps
        for j in 1:R
            # Generate Brownian increments for this realisation
            ΔW = sqrt(Δt)*randn(M,n)
            #initialise this realisation and simulate
            X[:,:,j] .= X0[:,j]
            for i in 1:n
                X[:,2,j] = X[:,2,j] + α(X[:,2,j],t[i],θ)*Δt + σ(X[:,2,j],t[i],θ)*ΔW[:,i]
            end
        end
        # return initial and final state
        return X,[t0,T],Δt
    end
end

@doc """
    Euler_Maruyama(α::Function, σ::Function, θ::Array{Float64,1}, T::Float64, t0::Float, X0::Array{Float64}, Δt::Float64, R::Int,paths=true) 

Simulates `R` realisations of a non-autonomous vector stochastic differential 
equation of the form

        dX_t = α(X_t,t;θ)dt + σ(X_t,t;θ)dW_t, 

where `{X_t,t ≥ 0}` is an N-dimensional continuous state Markov process,
`{W_t, t ≥ 0}` is an M-dimensional Wiener process, `α(⋅,⋅;θ) : R^N × T → R^N` 
is the drift function, and `σ(⋅,⋅; θ) R^N × T → R^(N×M)` is the diffusion function.
Both `α` and `σ` are parameterised by the vector `θ ∈ R^d`.

Inputs:\n
    `α` - drift function
    `σ` - diffusion function 
    `θ` - parameter vector
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
    `Δt` - timestep used in simulation, can  be smaller that the input value if 
           T is not an integer multiple of Δt
"""
function Euler_Maruyama(α::Function, σ::Function, θ::Array{Float64,1}, 
                        T::Float64, t0::Float64,X0::Array{Float64},Δt::Float64,
                        R::Int,paths=true)
    # time step
    n = Int(floor((T - t0)/Δt))
    # adjust Δt if n*Δt ≠ T
    n = ((T - t0) > n*Δt) ? n+1 : n
    return Euler_Maruyama(α, σ, θ, T, t0, X0, n, R, paths)
end
