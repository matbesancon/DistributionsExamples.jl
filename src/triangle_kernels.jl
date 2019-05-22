mt = Random.MersenneTwister(42)
(µ, σ) = (5.0, 1.0)
xs = [rand(mt, Normal(µ, σ)) for _ in 1:50]

ndist = Normal(0.0, 0.3)
gkernel = KernelDensity.kde(xs, ndist)

tdist = TriangularDist(-0.5, 0.5)
tkernel = KernelDensity.kde(xs, tdist)

p = Plots.plot(tkernel.x, tkernel.density, labels = "Triangular kernel")
Plots.plot!(p, gkernel.x, gkernel.density,
            labels = "Gaussian kernel", legend = :left)
Plots.title!(p, "Comparison of Gaussian and triangular kernels")
Plots.savefig(base_img * "triangle_kernel.jl")
