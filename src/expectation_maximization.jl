"""
    expectation_step(X, z, dists)

Computes the expectation step with:
- `X`: (m×n) multi-variate observations (rows are observations, columns are variables)
- `dists`: distributions, one for each category (unique label)
- `prior`: prior probability of each label
Returns `Z`: (m×n) matrix of probability distributions of each label for each observation
"""
function expectation_step(X, dists, prior)
    n = size(X, 1)
    Z = Matrix{Float64}(undef, n, length(prior))
    for k in eachindex(prior)
        for i in 1:n
            Z[i,k] = prior[k] * pdf(dists[k], X[i,:]) /
                     sum(
                        prior[j] * pdf(dists[j], X[i,:])
                        for j in eachindex(prior)
                    )
        end
    end
    return Z
end

"""
    maximization_step(::Type{<:D}, X, Z)

Maximization step, parameterized by the type of distribution used.
Returns a tuple (dists, π, loglike) with:
- `dists`: the created vector of distributions
- `prior` vector of updated prior probabilities of each label
"""
function maximization_step(::Type{<:MvNormal}, X, Z)
    n = size(X, 1)
    Nk = size(Z, 2)
    µ = map(1:Nk) do k
        num = sum(Z[i,k] .* X[i,:] for i in 1:n)
        den = sum(Z[i,k] for i in 1:n)
        num / den
    end

    Σ = map(1:Nk) do k
        num = zeros(size(X, 2), size(X, 2))
        for i in 1:n
            r = X[i,:] .- µ[k]
            num .= num .+ Z[i,k] .* (r * r')
        end
        den = sum(Z[i,k] for i in 1:n)
        num ./ den
    end
    prior = [inv(n) * sum(Z[i,k] for i in 1:n)
            for k in 1:Nk
        ]
    dists = map(1:Nk) do k
        MvNormal(µ[k], Σ[k] + 10e-7I)
    end
    return (dists, prior)
end

"""
    expectation_maximization(D::Type{<:Distribution}, X, Nk, maxiter::Integer = 500, loglike_diff = 10e-5; all_dists = nothing)

Complete EM algorithm, takes a type of distribution `D` used to
determine the maximization step, observations `X` and number of classes `Nk`.
"""
function expectation_maximization(D::Type{<:Distribution}, X, Nk, maxiter::Integer = 500, loglike_diff = 10e-5; all_dists = nothing)
    # initialize classes
    n = size(X,1)
    Z = zeros(n, Nk)
    for i in 1:n
        j0 = mod(i,Nk)+1
        j1 = j0 > 1 ? j0-1 : 2
        Z[i,j0] = 0.75
        Z[i,j1] = 0.25
    end
    (dists, prior) = maximization_step(D, X, Z)
    all_dists isa Vector && push!(all_dists, (dists, prior))
    l = loglike_mixture(X, dists, prior)
    lprev = 0.0
    iter = 0
    # EM iterations
    while iter < maxiter && abs(lprev-l) > loglike_diff
        Z = expectation_step(X, dists, prior)
        (dists, prior) = maximization_step(D, X, Z)
        all_dists isa Vector && push!(all_dists, (dists, prior))
        lprev = l
        l = loglike_mixture(X, dists, prior)
        iter += 1
    end
    return (dists, prior, Z, l, iter)
end

function loglike_mixture(X, dists, prior)
    l = zero(eltype(X))
    n = size(X,1)
    for i in 1:n
        l += log(
            sum(prior[k] * pdf(dists[k], X[i,:]) for k in eachindex(prior))
        )
    end
    return l
end

let
    Random.seed!(42)
    X = [randn(1000, 2); 3.0 .+ randn(1000, 2)]
    n = size(X, 1)
    for maxiter in (1, 5, 10, 100, 150, 250)
        (dists, prior, Z, l, iter) = expectation_maximization(MvNormal, X, 2, maxiter)
        contour_mixture(x1, x2) = maximum([pdf(dists[k], [x1,x2]) for k in 1:2])
        colors = map(1:n) do i
            (_, j) = findmax(Z[i,:])
            j == 1 ? :red : :blue
        end
        plt = Plots.contour(-2.8:0.05:6.0, -2.8:0.05:6.0, contour_mixture, contours = false, legend = nothing)
        Plots.scatter!(plt, X[:,1],X[:,2], label = "", color = colors)
        Plots.savefig(base_img * "mixture_iter$(maxiter)_$iter.pdf")
    end
end
