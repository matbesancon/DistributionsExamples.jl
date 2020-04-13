"""
Run some analyses on the wine dataset
"""

import DelimitedFiles

const wine_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
const wine_data_file = download(wine_data_url)

const raw_wine_data = DelimitedFiles.readdlm(wine_data_file,',', Float64)
const wine_quant = raw_wine_data[:,2:end]
const wine_labels = Int.(raw_wine_data[:,1])

Plots.scatter(wine_quant[:,1], wine_quant[:,2], group = wine_labels)
Plots.title!("Basic wine scatter plot")
Plots.savefig(joinpath(base_img, "basic_wine_scatter.pdf"))

# Fit a MLE on two first variables, second log-transformed
const res_normal = fit_mle(MvNormal, [wine_quant[:,1] log.(wine_quant[:,2])]')

# function for contour plot
function cont_func(x1, x2)
    pdf(res_normal, [x1,x2])
end

const wine_kde = KernelDensity.kde((wine_quant[:,1], log.(wine_quant[:,2])))

# contour plot for KDE
function cont_kde(x1, x2)
   pdf(wine_kde, x1, x2)
end

# local scope not to add plots to global one
let
    p = Plots.contour(11.0:0.05:15.0, -0.5:0.05:2.5, cont_func)
    Plots.scatter!(p, wine_quant[:,1], log.(wine_quant[:,2]), label = "Data points")
    Plots.title!(p, "Wine scatter plot & Gaussian maximum likelihood estimation")
    Plots.savefig(p, joinpath(base_img, "gaussian_contour_mle.pdf"))

    p = Plots.contour(11.0:0.05:15.0, -0.5:0.05:2.5, cont_kde)
    Plots.scatter!(p, wine_quant[:,1], log.(wine_quant[:,2]), group = wine_labels)
    Plots.title!(p, "Wine scatter plot & Kernel Density Estimation")
    Plots.savefig(p, joinpath(base_img, "gaussian_contour_kde.pdf"))
end

# fit a (Normal × LogNormal) on x1 × x2

function build_product_distribution(p)
    return Product([
        Normal(p[1], p[2]),
        LogNormal(p[3], p[4]),
    ])
end

function loglike(p)
    d = build_product_distribution(p)
    return loglikelihood(d.v[1], wine_quant[:,1]) + loglikelihood(d.v[2], wine_quant[:,2] .- 0.73)
end

∇L(p) = ForwardDiff.gradient(loglike, p)

let
    Random.seed!(42)
    # random start
    p = [10.0 + 3.0 * rand(), rand()+1, 2.0 + 3.0*rand(), rand()+1]
    iter = 1
    maxiter = 5000
    mean_paths = (Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter))
    while iter <= maxiter && sum(abs.(∇L(p))) >= 10^-6
        mean_paths[1][iter] = p[1]
        mean_paths[2][iter] = p[3]
        p = p + 0.05 * inv(iter+5) * ∇L(p)
        p[2] = p[2] < 0 ? -p[2] : p[2]
        p[4] = p[4] < 0 ? -p[4] : p[4]
        @info sum(abs.(∇L(p)))
        iter += 1
    end
    @info iter
    @info p
    d = build_product_distribution(p)
    cont_product(x1, x2) = pdf(d, [x1, x2])

    plt = Plots.contour(11.0:0.05:15.0, -0.5:0.05:2.5, cont_product, contours = false, legend = nothing)
    Plots.scatter!(plt, wine_quant[:,1], wine_quant[:,2] .- 0.73, legend = nothing)
    Plots.title!(plt, "Data points & final contour")
    Plots.xticks!(plt, 11.0:1.0:15.0)
    Plots.xaxis!(plt, (10.9,15.0))
    Plots.yaxis!(plt, (-0.5,5.2))
    plt1 = Plots.plot(collect(11.0:0.05:14.9), [pdf(d.v[1], x) for x in 11.0:0.05:14.9], legend = nothing, width=5, color = :blue)
    Plots.title!(plt1, "Normal marginal")
    Plots.xticks!(plt1, 11.0:1.0:15.0)
    Plots.xaxis!(plt1, (10.9,15.0))
    plt2 = Plots.plot([pdf(d.v[2], x) for x in 0.0:0.05:5.0], collect(0.0:0.05:5.0), legend = nothing, width=5, color = :red)
    Plots.yaxis!(plt2, (-0.5,5.2))
    Plots.title!(plt2, "Lognormal marginal")
    pfinal = Plots.plot(plt, plt2, plt1, Plots.plot(framestyle = :none), layout = (2,2), size = (1200,800))
    Plots.savefig(pfinal, joinpath(base_img, "wine_product_dist.pdf"))

    pltconv = Plots.scatter(wine_quant[:,1], wine_quant[:,2] .- 0.73, label = "Data points", markersize = 3)
    Plots.plot!(pltconv, mean_paths[1][1:iter-1], exp.(mean_paths[2][1:iter-1]), label = "", linestyle = :dot, linecolor = :black, alpha = 0.7)
    Plots.scatter!(pltconv, mean_paths[1][1:iter-1], exp.(mean_paths[2][1:iter-1]), label = "Mean iterations", marker = :cross, linecolor = :black, alpha = 0.8)
    Plots.scatter!(pltconv, [p[1]], [exp(p[3])], label = "Final mean", color = :red)
    Plots.title!(pltconv, "Likelihood maximization - $(iter-1) iterations")
    Plots.savefig(pltconv, joinpath(base_img, "wine_product_convergence.pdf"))
end

# reduced block for benchmark
function bench_func()
    # random start
    p = [10.0 + 3.0 * rand(), rand()+1, 2.0 + 3.0*rand(), rand()+1]
    iter = 1
    maxiter = 5000
    while iter <= maxiter && sum(abs.(∇L(p))) >= 10^-6
        p = p + 0.05 * inv(iter+5) * ∇L(p)
        p[2] = p[2] < 0 ? -p[2] : p[2]
        p[4] = p[4] < 0 ? -p[4] : p[4]
        iter += 1
    end
    return build_product_distribution(p)
end

let
    X = [wine_quant[:,1] log.(wine_quant[:,2])]
    n = size(X, 1)
    Nk = 3
    for maxiter in (0, 20, 22, 25, 30, 50, 60, 65, 67, 70, 80)
        (dists, prior, Z, l, iter) = expectation_maximization(MvNormal, X, Nk, maxiter)
        contour_mixture(x1, x2) = maximum([pdf(dists[k], [x1,x2]) for k in 1:Nk])
        colors = map(1:n) do i
            (_, j) = findmax(Z[i,:])
            j == 1 ? :red  :
            j == 2 ? :blue :
            :green
        end
        plt = Plots.contour(11.0:0.05:15.0, -0.5:0.05:2.5, contour_mixture, contours = false, label = "", size = (600,400))
        Plots.scatter!(plt, X[:,1],X[:,2], label = "", color = colors, markershape = :cross)
        Plots.scatter!(plt, [mean(d)[1] for d in dists], [mean(d)[2] for d in dists], color = :black, label = "Centroids")
        Plots.title!("Estimated mixture, $maxiter iterations")
        Plots.xlabel!("x1")
        Plots.ylabel!("x2")
        Plots.savefig(joinpath(base_img, "wine_mixture_iter$(maxiter)_$iter.pdf"))
    end
end
