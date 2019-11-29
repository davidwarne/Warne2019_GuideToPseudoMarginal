#=
# Functions to select predefined Stochastic (bio)chemical reaction network 
# models and generate sythetic data 
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#


@doc """
    SelectBCRN(name::String)

Shortcut function to create valid stoichiometric matrices and propensity functions
for common example Biochemical reaction networks. Example parameter values, initial 
conditions and simulation endtimes are also provided. The outputs from this function
ca be used directly in the `SimulateCLE` function.

Inputs:\n
    `name` - string that identifies the netork to return. Supported values are
             "prod-deg", "mono-mol-chain", "mich-ment", "schlogl", and "repressilator"
Outputs:\n
    `model` - LaTeX string for titles/labels/legend use in plots
    `species` - Array of strings containing chemical species labels
    `ν` - stoichiometric matrix ν ∈ R^(N×M) where
    `a` - propensity function a : R^N × R^Q → R^M
    `k` - example model parameters k ∈ R^Q, if k is ONLY the reaction rate 
          parameters then Q = M, but this need not be the case.
    `x0` - example initial condition X(0) = x0 ∈ R^N
    `T` - example simulation endtime

# Example
```julia-repl
    model,ν,a,k,x0,T = SelectBCRN("mono-mol-chain")
    R=1
    X0 = x0.*ones(1,R)
    X,t = SimulateCLE(a,ν,k,T,0.0,X0,0.001,R)
```
"""
function SelectBCRN(name::String)
    
    if name == "prod-deg" 
        # CLE for production-degradation ∅ ⇆ X
        model = "production-degradation"
        species = [L"$X$"]
        # model definition
        ν = [1.0 -1.0] 
        a = (X,k) -> [k[1], (X[1] <= 0.0) ? 0.0 : k[2]*X[1]] 
        # example parameterisation, initial condition and end time
        k = [1.0, 0.01] 
        x0 = [50.0] 
        T = 1000.0
        return model,species,ν,a,k,x0,T
    elseif name == "mono-mol-chain"
        # CLE for mono-molecular chain ∅ → A → B → ∅
        model = "mono-molecular chain"
        species = [L"$A$",L"$B$"]
        # model definition
        ν = [1.0 -1.0  0.0;
             0.0  1.0 -1.0]
        a = (X,k) -> [k[1]; 
                      (X[1] <= 0.0) ? 0.0 : k[2]*X[1]; 
                      (X[2] <= 0.0) ? 0.0 : k[3]*X[2]]
        # example parameterisation, initial condition and end time
        k = [1.0, 0.1, 0.05] 
        x0 = [100.0,0.0]
        T = 100.0
        return model,species,ν,a,k,x0,T
    elseif name == "mich-ment"
        # CLE for Michaelis-Menten enzyme kinetic model E + S ⇆ C → P
        model = "Michaelis-Menten"
        species = [L"$E$",L"$S$",L"$C$",L"$P$"]
        # model definition
        ν = [-1.0  1.0  1.0;
             -1.0  1.0  0.0;
              1.0 -1.0 -1.0;
              0.0  0.0  1.0]
        a = (X,k) -> [(X[1] <= 0.0 || X[2] <= 0.0) ? 0.0 : k[1]*X[1]*X[2];
                      (X[3] <= 0.0) ? 0.0 : k[2]*X[3];
                      (X[3] <= 0.0) ? 0.0 : k[3]*X[3]]
        #example parameterisation, initial condition and end tim
        k = [0.001, 0.005, 0.01]
        x0 = [100.0,100.0,0.0,0.0]
        T = 80.0
        return model,species,ν,a,k,x0,T
    elseif name == "schlogl"
        #  CLE for Schlogl model 2X ⇆ 3X, ∅ ⇆ X
        model = L"Schl\"{o}gl"
        species = [L"$X$"]
        # model definition
        ν = [1.0 -1.0 1.0 -1.0]
        a = (X,k) -> [(X[1] <= 1.0) ? 0.0 : k[1]*X[1]*(X[1]-1.0); 
                      (X[1] <= 2.0) ? 0.0 : k[2]*X[1]*(X[1] - 1.0)*(X[1] - 2.0);
                      k[3];
                      (X[1] <= 0.0) ? 0.0 : k[4]*X[1]]
        # example parameterisation, initial condition and end time
        k = [0.18, 2.5e-4, 2200.0, 37.5]
        x0 = [0.0]
        T = 100.0
        return model,species,ν,a,k,x0,T
    elseif name == "repressilator"
        # CLE for 3 gene repressilator network ∅ ⇆ m_i → m_i + p_i, p_i → ∅
        model = "repressilator}"
        species = [L"$M_1$",L"$P_1$",L"$M_2$",L"$P_2$",L"$M_3$",L"$P_3$"]
        # model definition
        ν = [1.0 -1.0 0.0  0.0  0.0  0.0 0.0  0.0 0.0  0.0 0.0  0.0; 
             0.0  0.0 1.0 -1.0  0.0  0.0 0.0  0.0 0.0  0.0 0.0  0.0; 
             0.0  0.0 0.0  0.0  1.0 -1.0 0.0  0.0 0.0  0.0 0.0  0.0; 
             0.0  0.0 0.0  0.0  0.0  0.0 1.0 -1.0 0.0  0.0 0.0  0.0; 
             0.0  0.0 0.0  0.0  0.0  0.0 0.0  0.0 1.0 -1.0 0.0  0.0;
             0.0  0.0 0.0  0.0  0.0  0.0 0.0  0.0 0.0  0.0 1.0 -1.0]
        a = (X,θ) -> [(X[6] <= 0.0) ? θ[2] + θ[1] : θ[2] + θ[1]/(1.0 + X[6]^θ[4]);
                      (X[1] <= 0.0) ? 0.0 : 1.0*X[1];
                      (X[1] <= 0.0) ? 0.0 : θ[3]*X[1];
                      (X[2] <= 0.0) ? 0.0 : θ[3]*X[2];
                      (X[2] <= 0.0) ? θ[2] + θ[1] : θ[2] + θ[1]/(1.0 + X[2]^θ[4]);
                      (X[3] <= 0.0) ? 0.0 : 1.0*X[3];
                      (X[3] <= 0.0) ? 0.0 : θ[3]*X[3];
                      (X[4] <= 0.0) ? 0.0 : θ[3]*X[4];
                      (X[4] <= 0.0) ? θ[2] + θ[1] : θ[2] + θ[1]/(1.0 + X[4]^θ[4]);
                      (X[5] <= 0.0) ? 0.0 : 1.0*X[5];
                      (X[5] <= 0.0) ? 0.0 : θ[3]*X[5];
                      (X[6] <= 0.0) ? 0.0 : θ[3]*X[6]]
        # example parameterisation, initial condition and end time
        α = 1000.0
        α0 = 1.0
        β = 5.0
        n = 2.0
        θ = [α, α0, β, n]
        x0 = [0.0,2.0,0.0,1.0,0.0,3.0]
        T = 50.0
        return model,species,ν,a,θ,x0,T
    else
        error("Unsupported model. See help?> SelectBCRN and help?> SimulateCLE")
    end
end

@doc """
    GenerateObservations(a::Function,ν::Array{Float64,2},θ::Array{Float64,1},x0::Array{Float64,1},Δt::Float64,t::Float64t::Array{Float64,1},g::Function)

Generates synthetic data for the Chemical reaction network defined by the stoichiometric matrix, `ν`,propensity function, `a`, parameterised by `θ` with initial condition `x0`. The Chemical Langevin SDE is solved numerically using the EUler-Maruyama scheme with step `Δt`. Observation error is modelled
using the function `g`.

Inputs:\n
    `a` - propensity function
    `ν` - stoichiometric matrix
    `θ` - parameter vector
    `x0` - initial condition
    `Δt` - step size for Euler-Maruyama scheme
    `t` - array of observation times
    `g` - observation process
Outputs:\n
    `Y` - Array of observations. `Y[:,j] = g(X[:,j])` where `X[:,j]` is the CLE solution at time `t[j]` 
"""
function GenerateObservations(a::Function,ν::Array{Float64,2},θ::Array{Float64,1},x0::Array{Float64,1},Δt::Float64,t::Array{Float64,1},g::Function)

    # check dimensions
    T = length(t)
    M = length(g(x0)) # |y0| ≤ |x0|

    # allocate memory for obserations
    Y = zeros(M,T)
    Y[:,1] = g(x0)
    Xs = x0.*ones(1,1)
    for j in 2:T
        X, = SimulateCLE(a,ν,θ,t[j],t[j-1],Xs,Δt,1,false)
        Y[:,j] = g(X[:,2,1])
        Xs = X[:,2,:]
    end
    return Y
end
