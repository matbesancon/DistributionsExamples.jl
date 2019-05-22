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
