xvals = 0.01:0.01:8.0
yvals = map(xvals) do x
    comp1 = pdf(LogNormal(), x)
    comp2 = pdf(Uniform(2.0, 3.0), x)
    0.5 * comp1 + 0.5 * comp2
end

p = Plots.plot()
Plots.plot!(p, xvals, yvals, labels = "Real distribution")

kde = KernelDensity.kde(xs, bandwidth = 0.1)
Plots.plot!(p, kde.x, kde.density, labels = "KDE")

Plots.plot!(p, xvals, yvals, labels = "Real distribution")
Plots.xlims!(p, 0.0, 8.0)
Plots.savefig(base_img * "compare_kde.pdf")
