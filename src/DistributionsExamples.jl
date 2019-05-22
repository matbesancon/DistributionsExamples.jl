module DistributionsExamples

import Random
import Plots
import KernelDensity
using Distributions
import ForwardDiff
using LinearAlgebra: dot, I

const base_img = (@__DIR__) * "/../img/"

include("kde_gaussian.jl")
include("compare_kde.jl")
include("triangle_kernels.jl")
include("gaussian_mm.jl")
include("expectation_maximization.jl")
include("wine.jl")

end # module
