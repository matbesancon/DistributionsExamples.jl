# Plot reproduction folder

## How to use

From the Julia REPL, in this folder:

```julia
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
julia> import DistributionsExamples
```

The first three command ensure reproducibility by freezing all package
versions. The last one executes each of the scripts in `/src` and
produces the images in `/img`.

These examples were used for the [paper](https://arxiv.org/abs/1907.08611) introducing the Distributions.jl package. 
