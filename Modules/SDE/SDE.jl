"""
A Julia module for numerical solutions to stochastic differential equations

Supported numerical schemes:
    Euler-Maruyama 
"""
module SDE 
#=
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

using LaTeXStrings
using LinearAlgebra

export Euler_Maruyama
export SimulateCLE
export SelectBCRN
export GenerateObservations

include("EulerMaruyama.jl")
include("ChemicalLangevin.jl")
include("ChemicalReactionNetworkModels.jl")

end
