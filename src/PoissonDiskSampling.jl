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
    r::Float64
    min::NTuple{dim, Float64}
    max::NTuple{dim, Float64}
    size::NTuple{dim, Int}
end
Base.size(grid::Grid) = grid.size
@inline sampling_distance(grid::Grid) = grid.r

function Grid(dx::Real, minmaxes::Vararg{Tuple{Real, Real}, n}) where {n}
    all(minmax->minmax[1]<minmax[2], minmaxes) || throw(ArgumentError("`(min, max)` must be `min < max`"))
    axes = map(minmax->minmax[1]:dx:minmax[2], minmaxes)
    min = map(first, axes)
    max = map(last, axes)
    Grid{n}(dx, dx*√n, min, max, map(length, axes))
end

function whichcell(x::Vec, grid::Grid)
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
    cells = fill(nanvec(Vec{dim, Float64}), size(grid).-1)
    generate!(rng, cells, grid, num_generations)
    filter!(!isnanvec, vec(cells))
end

function generate!(rng, cells::Array, grid::Grid{dim}, num_generations::Int) where {dim}
    active_list = CartesianIndex{dim}[]

    push!(active_list, set_point!(cells, random_point(rng, grid), grid))

    while !isempty(active_list)
        index = rand(rng, 1:length(active_list))
        xᵢ = cells[active_list[index]]
        found = false
        for k in 1:num_generations
            r = sampling_distance(grid)
            Iₖ = set_point!(cells, random_point(rng, Annulus(xᵢ, r, 2r)), grid)
            if Iₖ !== nothing
                push!(active_list, Iₖ)
                found = true
            end
        end
        !found && deleteat!(active_list, index)
    end

    cells
end

function set_point!(cells, xₖ, grid)
    Iₖ = whichcell(xₖ, grid)
    Iₖ === nothing && return nothing
    u = 2*oneunit(Iₖ)
    neighborcells = CartesianIndices(cells) ∩ ((Iₖ-u):(Iₖ+u))
    valid = all(neighborcells) do cellindex
        x = cells[cellindex]
        isnanvec(x) && return true
        sum(abs2, xₖ .- x) > abs2(sampling_distance(grid))
    end
    if valid
        @assert isnanvec(cells[Iₖ])
        cells[Iₖ] = xₖ
        return Iₖ
    end
    nothing
end

@inline nanvec(::Type{Vec{dim, T}}) where {dim, T} = Vec{dim, T}(ntuple(i->NaN, Val(dim)))
@inline isnanvec(x::Vec) = x === nanvec(typeof(x))

end # module PoissonDiskSampling
