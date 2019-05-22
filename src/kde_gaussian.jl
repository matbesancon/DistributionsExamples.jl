function generate_point(rng = Random.GLOBAL_RNG)
    thres = rand(rng)
    if thres >= 0.5
        rand(rng, LogNormal())
    else  
        rand(rng, Uniform(2.0, 3.0))
    end
end

mt = Random.MersenneTwister(42)
xs = [generate_point(mt) for _ in 1:5000]
bandwidths = [0.05, 0.1, 0.5]
densities = [KernelDensity.kde(xs, bandwidth = bw) for bw in bandwidths]

p = Plots.plot()
for (b,d) in zip(bandwidths, densities)
    Plots.plot!(p, d.x, d.density, labels = "bw = $b")
end
Plots.xlims!(p, 0.0, 8.0)
Plots.title!("KDE with Gaussian Kernel")
Plots.savefig(base_img * "KDE.pdf")
