# performance PDF evaluation

using BenchmarkTools: @belapsed, @btime
using Distributions
import Random

distributions = [
    Normal(-1.0, 0.3),
    Normal(0.0, 0.5),
    Normal(3.0, 1.0),
]

priors = [0.25, 0.25, 0.5]

gmm_normal = MixtureModel(distributions, priors)

function manual_computation(mixture, x)
    v = zero(eltype(mixture))
    p = probs(mixture)
    d = components(mixture)
    for i in 1:ncomponents(mixture)
        if p[i] > 0
            v += p[i] * pdf(d[i], x)
        end
    end
    return v
end

@info "Small"

Random.seed!(42)
for x in rand(5)
    bauto = @belapsed pdf($gmm_normal, $x)
    bmanual = @belapsed manual_computation($gmm_normal, $x)
    @btime manual_computation($gmm_normal, $x)
    @show bauto
    @show bmanual
    @show bauto / bmanual
end

large_normals = [Normal(rand(), rand()) for _ in 1:1000]
large_probs = [rand() for _ in 1:1000]
large_probs .= large_probs ./ sum(large_probs)

gmm_large = MixtureModel(large_normals, large_probs)

@info "Large"

Random.seed!(42)
for x in rand(5)
    bauto = @belapsed pdf($gmm_large, $x)
    bmanual = @belapsed manual_computation($gmm_large, $x)
    @btime manual_computation($gmm_large, $x)
    @show bauto
    @show bmanual
    @show bauto / bmanual
end

# heterogenerous distributions

large_het = append!(
    ContinuousUnivariateDistribution[Normal(rand(), rand()) for _ in 1:1000],
    ContinuousUnivariateDistribution[LogNormal(rand(), rand()) for _ in 1:1000],
)

large_het_probs = [rand() for _ in 1:2000]
large_het_probs .= large_het_probs ./ sum(large_het_probs)

gmm_het = MixtureModel(large_het, large_het_probs)

@info "Heterogenerous"

Random.seed!(42)
for x in rand(5)
    bauto = @belapsed pdf($gmm_het, $x)
    bmanual = @belapsed manual_computation($gmm_het, $x)
    @btime manual_computation($gmm_het, $x)
    @show bauto
    @show bmanual
    @show bauto / bmanual
end
