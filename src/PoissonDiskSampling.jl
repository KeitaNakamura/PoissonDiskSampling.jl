module PoissonDiskSampling

using Random

const Vec{dim, T} = NTuple{dim, T}

"""
    Grid(dx, (min_1, max_1)..., (min_n, max_n))

Construct grid with the domain ``[min_1, max_1)`` ... ``[min_n, max_n)``,
where ``max_n`` could be changed due to the grid spacing `dx`.
"""
struct Grid{dim}
    dx::Float64
    min::NTuple{dim, Float64}
    max::NTuple{dim, Float64}
    size::NTuple{dim, Int}
end
Base.size(grid::Grid) = grid.size

function Grid(dx::Real, minmaxes::Vararg{Tuple{Real, Real}, dim}) where {dim}
    all(minmax->minmax[1]<minmax[2], minmaxes) || throw(ArgumentError("`(min, max)` must be `min < max`"))
    axes = map(minmax->minmax[1]:dx:minmax[2], minmaxes)
    min = map(first, axes)
    max = map(last, axes)
    Grid{dim}(dx, min, max, map(length, axes))
end

function whichcell(grid::Grid, x::Vec)
    xmin = grid.min
    dx⁻¹ = inv(grid.dx)
    ξ = @. (x - xmin) * dx⁻¹
    ncells = size(grid) .- 1
    all(@. 0 ≤ ξ < ncells) || return nothing # use `<` because of `floor`
    CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

function random_point(rng, grid::Grid)
    map(grid.min, grid.max) do xmin, xmax
        xmin + rand(rng) * (xmax - xmin)
    end
end

sampling_distance(grid::Grid{n}) where {n} = grid.dx * √n

struct Annulus{dim}
    centroid::Vec{dim, Float64}
    r1::Float64
    r2::Float64
end

function random_point(rng, annulus::Annulus{n}) where {n}
    n > 1 || throw(ArgumentError("dimensions must be ≥ 2"))
    r = annulus.r1 + rand(rng) * (annulus.r2 - annulus.r1)
    θs = ntuple(i->rand(rng)*ifelse(i==n-1, 2π, π), Val(n-1))
    map(+, annulus.centroid, spherical_coordinates(r, θs))
end

# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
@inline function spherical_coordinates(r::Float64, θs::Tuple{Vararg{Float64}})
    broadcast(*, (r,), coords_sin(θs), coords_cos(θs))
end
@inline coords_sin(::Tuple{}) = (1.0,)
@inline function coords_sin(θs::Tuple{Vararg{Float64}})
    xs = coords_sin(Base.front(θs))
    xlast = xs[end] * sin(θs[end])
    (xs..., xlast)
end
@inline function coords_cos(θs::Tuple{Vararg{Float64}})
    (map(cos, θs)..., 1.0)
end

"""
    PoissonDiskSampling.generate(r, (min_1, max_1)..., (min_n, max_n); k = 30)

Geneate coordinates based on the Poisson disk sampling.

The domain must be rectangle as ``[min_1, max_1)`` ... ``[min_n, max_n)``.
`r` is the minimum distance between samples. `k` is the number of trials for sampling at
each smaple, i.e., the algorithm will give up if no valid sample is found after `k` trials.

See *https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf* for more details.
"""
function generate(r::Real, minmaxes::Vararg{Tuple{Real, Real}}; k::Int=30)
    generate(Random.GLOBAL_RNG, r, minmaxes...; k)
end
function generate(rng, r, minmaxes::Vararg{Tuple{Real, Real}, n}; k::Int=30) where {n}
    generate(rng, Grid(r/√n, minmaxes...), k)
end

# https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
function generate(rng, grid::Grid{dim}, num_generations::Int) where {dim}
    r = sampling_distance(grid)
    cells = zeros(UInt, size(grid).-1)

    active_list = Int[]
    points = Vec{dim, Float64}[]

    push!(points, random_point(rng, grid))
    push!(active_list, length(points))
    cells[whichcell(grid, points[end])] = length(points)

    while !isempty(active_list)
        index = rand(rng, 1:length(active_list))
        x = points[active_list[index]]
        found = false
        for k in 1:num_generations
            x_k = random_point(rng, Annulus(x, r, 2r))
            I = whichcell(grid, x_k)
            I === nothing && continue
            u = 2*oneunit(I)
            neighborcells = CartesianIndices(cells) ∩ ((I-u):(I+u))
            valid = all(neighborcells) do cellid
                ptid = cells[cellid]
                iszero(ptid) && return true
                p = points[ptid]
                sum(abs2, x_k .- p) > abs2(r)
            end
            if valid
                @assert iszero(cells[I])
                push!(points, x_k)
                push!(active_list, length(points))
                cells[I] = length(points)
                found = true
            end
        end
        !found && deleteat!(active_list, index)
    end

    points
end

end # module PoissonDiskSampling
