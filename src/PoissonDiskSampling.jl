module PoissonDiskSampling

using Base.Iterators: product
using Random

const BLOCKFACTOR = unsigned(3) # 2^3
const Vec{dim, T} = NTuple{dim, T}

"""
    Grid(r, (min_1, max_1)..., (min_n, max_n))

Construct grid with the domain ``[min_1, max_1)`` ... ``[min_n, max_n)``,
where ``max_n`` could be changed based on the minimum distance `r` between samples.
"""
struct Grid{dim}
    r::Float64
    dx::Float64
    min::NTuple{dim, Float64}
    max::NTuple{dim, Float64}
    size::NTuple{dim, Int}
    offset::CartesianIndex{dim}
end
Base.size(grid::Grid) = grid.size
@inline sampling_distance(grid::Grid) = grid.r

function Grid(r::Real, minmaxes::Vararg{Tuple{Real, Real}, n}) where {n}
    all(minmax->minmax[1]<minmax[2], minmaxes) || throw(ArgumentError("`(min, max)` must be `min < max`"))
    dx = r/√n
    axes = map(minmax->minmax[1]:dx:minmax[2], minmaxes)
    Grid{n}(r, dx, map(first, axes), map(last, axes), map(length, axes), zero(CartesianIndex{n}))
end

function whichcell(x::Vec, grid::Grid)
    xmin = grid.min
    dx⁻¹ = inv(grid.dx)
    ξ = @. (x - xmin) * dx⁻¹
    ncells = size(grid) .- 1
    all(@. 0 ≤ ξ < ncells) || return nothing # use `<` because of `floor`
    grid.offset + CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

function random_point(rng, grid::Grid)
    map(grid.min, grid.max) do xmin, xmax
        xmin + rand(rng) * (xmax - xmin)
    end
end

function partition(grid::Grid{dim}, CI::CartesianIndices{dim}) where {dim}
    @boundscheck checkbounds(CartesianIndices(size(grid)), CI)
    Imin = first(CI).I
    Imax = last(CI).I
    new_min = @. grid.min + grid.dx * (Imin - 1)
    new_max = @. grid.min + grid.dx * (Imax - 1)
    Grid(grid.dx, grid.r, new_min, new_max, size(CI), CartesianIndex(Imin .- 1))
end

# block methods
blocksize(gridsize::Tuple{Vararg{Int}}) = @. (gridsize-1)>>BLOCKFACTOR+1
blocksize(grid::Grid) = blocksize(size(grid))
function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = product(ntuple(i->1:2, Val(dim))...)
    vec(map(st -> map(CartesianIndex{dim}, Iterators.product(StepRange.(st, 2, blocksize)...))::Array{CartesianIndex{dim}, dim}, starts))
end
function gridindices_from_blockindex(grid::Grid, blk::CartesianIndex)
    start = @. (blk.I-1) << BLOCKFACTOR + 1
    stop = @. start + (1 << BLOCKFACTOR)
    (CartesianIndex(start):CartesianIndex(stop)) ∩ CartesianIndices(size(grid))
end
function blockpartition(grid::Grid{dim}, blk::CartesianIndex{dim}) where {dim}
    @boundscheck checkbounds(CartesianIndices(blocksize(grid)), blk)
    partition(grid, gridindices_from_blockindex(grid, blk))
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
    PoissonDiskSampling.generate(r, (min_1, max_1)..., (min_n, max_n); k = 30, parallel = true)

Geneate coordinates based on the Poisson disk sampling.

The domain must be rectangle as ``[min_1, max_1)`` ... ``[min_n, max_n)``.
`r` is the minimum distance between samples. `k` is the number of trials for sampling at
each smaple, i.e., the algorithm will give up if no valid sample is found after `k` trials.
If `Threads.nthreads() > 1 && parallel`, multithreading is enabled.

See *https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf* for more details.
"""
function generate(r::Real, minmaxes::Vararg{Tuple{Real, Real}}; k::Int=30, parallel::Bool=true)
    generate(Random.GLOBAL_RNG, r, minmaxes...; k, parallel)
end
function generate(rng, r, minmaxes::Vararg{Tuple{Real, Real}}; k::Int=30, parallel::Bool=true)
    generate(rng, Grid(r, minmaxes...), k, parallel)
end

function generate(rng, grid::Grid{dim}, num_generations::Int, parallel::Bool) where {dim}
    cells = fill(nanvec(Vec{dim, Float64}), size(grid).-1)
    if parallel && Threads.nthreads() > 1
        for blocks in threadsafe_blocks(blocksize(grid))
            Threads.@threads :static for blk in blocks
                generate!(rng, cells, grid, gridindices_from_blockindex(grid, blk), num_generations)
            end
        end
    else
        generate!(rng, cells, grid, CartesianIndices(size(grid)), num_generations)
    end
    filter!(!isnanvec, vec(cells))
end

# https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
function generate!(rng, cells::Array, grid::Grid{dim}, gridindices::CartesianIndices, num_generations::Int) where {dim}
    partgrid = partition(grid, gridindices)
    cellindices = first(gridindices):(last(gridindices) - oneunit(CartesianIndex{dim}))

    active_list = filter(i->!isnanvec(cells[i]), cellindices)
    if isempty(active_list)
        found = false
        for k in 1:num_generations*2
            I₀ = set_point!(cells, random_point(rng, partgrid), grid)
            if I₀ !== nothing
                push!(active_list, I₀)
                found = true
                break
            end
        end
        !found && return cells
    end

    while !isempty(active_list)
        index = rand(rng, 1:length(active_list))
        xᵢ = cells[active_list[index]]
        found = false
        for k in 1:num_generations
            r = sampling_distance(grid)
            Iₖ = set_point!(cells, random_point(rng, Annulus(xᵢ, r, 2r)), grid)
            if Iₖ !== nothing
                # `Iₖ` can be outside of `partgrid`, but add `Iₖ` into `active_list`
                # only when it is in `partgrid`.
                if Iₖ in cellindices
                    push!(active_list, Iₖ)
                end
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
