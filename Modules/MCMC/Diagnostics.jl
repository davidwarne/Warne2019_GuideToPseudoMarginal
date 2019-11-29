#=
# Functions for the evaluation of MCMC convergence. Specifically, we implement the
# rank normalised split-\hat{R} and folded-split-\hat{R} and the between chain 
# effective sample size S_eff of Vehtari, Gelman, Simpson, Carpenter, and Burkner.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

@doc """
    Seff(θ:Array{Array{Float64,2},1},W::Array{Float64},varhat::Array{Float64,1})

Effective Sample Size as defined by Gelman et al., BDA 3rd edn.

"""
function Seff(θ::Array{Array{Float64,2},1},W::Array{Float64},varhat::Array{Float64,1})
    M = length(θ)
    d,N = size(θ[1])
    S = M*N
    
    ρ = deepcopy(θ)
    τ = zeros(d)
    # compute auto-correlation function for all chains at parameter k
    for j in 1:M
        ρ[j] = collect(autocor(θ[j]',0:N-1)')
    end
    for k in 1:d
        ρhat = zeros(1,N)
        for j in 1:M
            ρhat[:] += ρ[j][k,:]
        end
        ρhat[:] ./= M
        ρhat[:] = W[k] .- ρhat[:]
        ρhat[:] ./= varhat[k] 
        ρhat[:] = 1 .- ρhat[:]
        T = 1
        while T < N-3 && ρhat[T+1]+ρhat[T+2] > 0.0
            T += 2
        end
        τ[k] = 1 + 2*sum(ρhat[1:T])
    end
    return S ./ τ
end

@doc """
    splitRhat(θ::Array{Array{Float64,2},1})

Computes the `split-\\hat{R}` statistic for M chains each contianing N samples 
and may take values in R^d. θ[j] is the jth chain, θ[j][i,:] is the trace of the
ith dimension of the jth chain and θ[j][:,k] is the kth sample of the jth chain.

Inputs:\n
    `θ` - Array of M chains, each having N samples taking values in R^d.
Outputs:\n
    `Rhat` - an array of d \\hat{R} values (2 per chain half per parameter)
"""
function splitRhat(θ::Array{Array{Float64,2},1})

    # get dimensions
    M = length(θ)
    d,N = size(θ[1])

    # compute \hat{R} for each parameter
    Rhat = zeros(d)
    W = zeros(d)
    B = zeros(d)
    varhat = zeros(d)
    for k in 1:d
        θbar = zeros(2*M)
        ssqrd = zeros(2*M)
        # compute within-chain variance
        for j in 1:M
            # first half of chain
            θbar[j] = 2*sum(θ[j][k,1:Int(N/2)])/N
            ssqrd[j] = 2*sum((θ[j][k,1:Int(N/2)] .- θbar[j]).^2)/(N-2)
            # second half of chain
            θbar[j+M] = 2*sum(θ[j][k,(Int(N/2)+1):N])/N
            ssqrd[j+M] = 2*sum((θ[j][k,(Int(N/2)+1):N] .- θbar[j]).^2)/(N-2)
        end
        W[k] = sum(ssqrd)/(2*M)
        
        # compute between-chain variance
        Eθbar = sum(θbar)/(2*M)
        B[k] = (N/(4*M - 2))*sum((θbar .- Eθbar).^2)
        
        # estimate of marginal posterior variance
        varhat[k] = ((N-2)/N)*W[k] + (2/N)*B[k]
        Rhat[k] = sqrt(varhat[k]/W[k])
    end
    ESS = Seff(θ,W,varhat)
    return Rhat,W,varhat,ESS
end

@doc """
    rnRhat(θ::Array{Array{Float64,2},1})

Computes the rank-normalised `\\hat{R}` statistic for M chains with N samples taking
values in R^d. This statistic should be < 1.01 for all dimensions d (along with 
appropriate measures of effective sample size).  

Inputs:\n
    `θ` - Array of M chains, each having N samples taking values in R^d.
Outputs:\n
    `Rhat` - an array of d \\hat{R} values (2 per chain half per parameter)
"""
function rnRhat(θ::Array{Array{Float64,2},1})
    M = length(θ)
    d,N = size(θ[1])
    S = M*N
    # compute pooled rank r^{i,j} for each θ[j][:,i]
    r = deepcopy(θ)
    z = deepcopy(θ)
    for k in 1:d
        p = []
        # pool samples for this dimension
        for j in 1:M
            p = vcat(p,θ[j][k,:])
        end
        # compute pooled ranks
        I = sortperm(p)
        p[I] = collect(1:S) # avoid allocating more memory
        # pack into r 
        for j in 1:M
            r[j][k,:] = p[((j-1)*N+1):j*N]
        end
    end
    # compute normalised rank transform
    for k in 1:d
        for j in 1:M
            z[j][k,:] = invlogcdf.(Normal(0.0,1.0),log.((r[j][k,:] .- 0.5)./S))
        end
    end
    return splitRhat(z)
end

