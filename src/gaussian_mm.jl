gmm = MixtureModel(
    Normal.([-1.0, 0.0, 3.0], # mean vector
            [0.3, 0.5, 1.0]),  # std vector
    [0.25, 0.25, 0.5] # component weights
)

xs  = -2.0:0.01:6.0
Plots.plot(xs, pdf.(gmm, xs), legend=nothing)
Plots.ylabel!("\$f_X(x)\$")
Plots.xlabel!("\$x\$")
Plots.title!("Gaussian mixture PDF")
Plots.savefig(base_img * "gaussian_mm.pdf")
