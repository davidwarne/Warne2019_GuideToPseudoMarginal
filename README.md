# Julia Code Examples of Pseuo-marginal methods fro computational inferens

This repository contains useful example Julia functions and scripts as an introduction to 
pseudo-marginal methods for inference on biochemical reaction networks using the
chemical Langevin formulation.

## Developer

David J. Warne (david.warne@qut.edu.au),
                School of Mathematical Sciences, 
                Science and Engineering Faculty, 
                Queensland Univeristy of Technology 
                
Google Scholar: (https://scholar.google.com.au/citations?user=t8l-kuoAAAAJ&hl=en)

## Citation Information

This code is provided as supplementary information to the paper,

David J Warne, Ruth E Baker, and Matthew J Simpson. A practical guide to 
pseudo-marginal methods for computational inference. ArXiv pre-print, TBA. 

## Licensing
This source code is licensed under the GNU General Public License Version 3.
Copyright (C) 2019 David J. Warne

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Contents

This folder contains a number of instructive Julia implementations 
for Stochastic simulation and computational inference. Demonstration scripts
showing typical usage are also provided
```bash
The directory structure is as follows
|-- Modules
    |-- SDE
        |-- SDE.jl
        |-- ChemicalReactionNetworkModels.jl
        |-- EulerMaruyama.jl
        |-- ChemicalLangevin.jl
    |-- MCMC
        |-- MCMC.jl
        |-- MetropolisHastings.jl
        |-- ABCMCMC.jl
        |-- PseudoMarginalMetropolisHastings.jl
        |-- BootstrapParticleFilter.jl
        |-- Diagnostics.jl
|-- Demonstrations
    |-- DemoProdDeg.jl
    |-- DemoMichMent.jl
    |-- DemoSchlogl.jl
    |-- DemoMH.jl
    |-- DemoABCMCMC.jl
    |-- DemoPMMH.jl
    |-- DemoMichMentPMCMC.jl
    |-- DemoSchloglPMCMC.jl
    |-- plotParticleFilter.jl
    |-- plotMichMentMCMCconv.jl
    |-- plotSchloglMCMCconv.jl
```

## Requirements

The following list contains the Julia version and required Modules used by this
project. Older or newer versions may work, but this has not been tested.

1. Julia            version 1.2.0
2. PyPlot           version 2.8.2
3. LaTeXStrings     version 1.0.3
4. Distributions    version 0.21.1
5. StatsBase        version 0.32.0
6. QuadGK           version 2.1.0
7. KernelDensity    version 0.5.1
8. JLD2             version 0.1.3

## Usage

Follow these steps to run the demonstrations:

1. Browse to repository folder
2. Add `Modules/SDE/` and `Modules/MCMC/` to the `JULIA_LOAD_PATH` environment
   variable. For example, in `bash`
   `[jbloggs@localhost]$ export JULIA_LOAD_PATH=./Modules/MCMC/:./Modules/SDE/:$JULIA_LOAD_PATH`
2. Start Julia, e.g.,
   `[jbloggs@localhost]$ julia`
3. To run a demo, use the Julia command prompt (REPL, read-execute-print loop), e.g.,
   `julia> include("./Demonstrations/DemoProdDeg.jl")` 

## List of examples

The following list of examples shows how to reproduce the figures in the main paper. For more computationlly intensive examples approximate run times are given for an Intel(R) Core(TM) i7-5600U CPU (2.6 GHz).

### Figure 1

    `julia> include("./Demonstrations/DemoProdDeg.jl")` 

### Figures 2 and 3

    `julia> include("./Demonstrations/DemoMH.jl")` 

### Figure 4

For Figure 4(A)--(F)
    `julia> include("./Demonstrations/DemoABCMCMC.jl")` 
For Figure 4(G)--(L)
    `julia> include("./Demonstrations/DemoPMMH.jl")` 

### Figure 5

    `julia> include("./Demonstrations/plotParticleFilter.jl")` 

### Figure 6

    `julia> include("./Demonstrations/DemoMichMent.jl")` 

### Figure 7 and 8

Generate Markov Chain trajectories (Warning: run time approx. 2 hrs2 hrs) 
    `julia> include("./Demonstrations/DemoMichMentPMCMC.jl")` 

Generate figures using output data file 
    `julia> include("./Demonstrations/plotMichMentMCMCconv.jl")` 

### Figure 9

    `julia> include("./Demonstrations/DemoSchlogl.jl")` 

### Figure 10 and 11

Generate Markov Chain trajectories (Warning: run time approx. 48 hrs ) 
    `julia> include("./Demonstrations/DemoSchloglPMCMC.jl")` 

Generate figures using output data file 
    `julia> include("./Demonstrations/plotSchloglMCMCconv.jl")` 
