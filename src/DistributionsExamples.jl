module DistributionsExamples

import Random
import Plots
import KernelDensity
using Distributions
import ForwardDiff
using LinearAlgebra: dot, I

const base_img = joinpath(@__DIR__, "../Figures/")

const scripts = 

const scripts = [
    "kde_gaussian.jl",
    "compare_kde.jl",
    "triangle_kernels.jl",
    "gaussian_mm.jl",
    "expectation_maximization.jl",
    "wine.jl",
]

for script in scripts
    @info "Running $script"
    include(script)
end

end # module
