module PoissonDiskSampling

using Base.Iterators: product
using Random
using StableRNGs

const BLOCKFACTOR = unsigned(3) # 2^3
const Vec{dim, T} = NTuple{dim, T}

"""
    Grid(r, (min_1, max_1)..., (min_n, max_n))

Construct grid with the domain ``[min_1, max_1)`` ... ``[min_n, max_n)``,
where ``max_n`` could be changed based on the minimum distance `r` between samples.
"""
struct Grid{dim, T}
    r::T
    dx::T
    min::NTuple{dim, T}
    max::NTuple{dim, T}
    size::NTuple{dim, Int}
    offset::CartesianIndex{dim}
end
Base.size(grid::Grid) = grid.size
@inline sampling_distance(grid::Grid) = grid.r

function Grid(::Type{T}, r::Real, minmaxes::Vararg{Tuple{Real, Real}, n}) where {T, n}
    all(minmax->minmax[1]<minmax[2], minmaxes) || throw(ArgumentError("`(min, max)` must be `min < max`"))
    dx = r/√n
    axes = map(minmax->minmax[1]:dx:minmax[2], minmaxes)
    Grid{n, T}(r, dx, map(first, axes), map(last, axes), map(length, axes), zero(CartesianIndex{n}))
end

function whichcell(x::Vec{dim, T}, grid::Grid{dim, T}) where {dim, T}
    xmin = grid.min
    dx⁻¹ = inv(grid.dx)
    ξ = @. (x - xmin) * dx⁻¹
    ncells = size(grid) .- 1
    all(@. 0 ≤ ξ < ncells) || return nothing # use `<` because of `floor`
    grid.offset + CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

function random_point(rng, grid::Grid{dim, T}) where {dim, T}
    map(grid.min, grid.max) do xmin, xmax
        xmin + rand(rng, T) * (xmax - xmin)
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

struct Annulus{dim, T}
    centroid::Vec{dim, T}
    r1::T
    r2::T
end

function random_point(rng, annulus::Annulus{n, T}) where {n, T}
    n > 1 || throw(ArgumentError("dimensions must be ≥ 2"))
    r = annulus.r1 + rand(rng, T) * (annulus.r2 - annulus.r1)
    θs = ntuple(i -> rand(rng, T) * T(ifelse(i==n-1, 2π, π)), Val(n-1))
    map(+, annulus.centroid, spherical_coordinates(r, θs))
end

# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
@inline function spherical_coordinates(r::T, θs::Tuple{Vararg{T}}) where {T}
    broadcast(*, (r,), coords_sin(θs), coords_cos(θs))
end
@inline coords_sin(::Tuple{}) = (1,)
@inline function coords_sin(θs::Tuple{Vararg{T}}) where {T}
    xs = coords_sin(Base.front(θs))
    xlast = xs[end] * sin(θs[end])
    (xs..., xlast)
end
@inline function coords_cos(θs::Tuple{Vararg{T}}) where {T}
    (map(cos, θs)..., one(T))
end

"""
    PoissonDiskSampling.generate([rng=GLOBAL_RNG], [T=Float64], r, (min_1, max_1)..., (min_n, max_n); k = 30, parallel = true)

Geneate points based on the Poisson disk sampling.

The domain must be rectangle as ``[min_1, max_1)`` ... ``[min_n, max_n)``.
`r` is the minimum distance between samples. `k` is the number of trials for sampling,
i.e., the algorithm will give up if no valid sample is found after `k` trials.
If `Threads.nthreads() > 1 && parallel`, multithreading is enabled.

The algorithm is based on *https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf*.
"""
generate(args...; k::Int=30, parallel::Bool=true) = _generate(args...; k, parallel)
_generate(rng, ::Type{T}, r, minmaxes::Tuple{Real, Real}...; k, parallel) where {T} = generate(rng, Grid(T, r, minmaxes...), k, parallel)
_generate(rng,            r, minmaxes::Tuple{Real, Real}...; k, parallel)           = _generate(rng,               Float64, r, minmaxes...; k, parallel)
_generate(     ::Type{T}, r, minmaxes::Tuple{Real, Real}...; k, parallel) where {T} = _generate(Random.GLOBAL_RNG, T,       r, minmaxes...; k, parallel)
_generate(                r, minmaxes::Tuple{Real, Real}...; k, parallel)           = _generate(Random.GLOBAL_RNG,          r, minmaxes...; k, parallel)

get_threads_rngs(rng) = fill(rng, Threads.nthreads())
get_threads_rngs(rng::StableRNG) = [copy(rng) for _ in 1:Threads.nthreads()]
function generate(rng, grid::Grid{dim, T}, num_generations::Int, parallel::Bool) where {dim, T}
    cells = fill(nanvec(Vec{dim, T}), size(grid).-1)
    if parallel && Threads.nthreads() > 1
        rngs = get_threads_rngs(rng)
        for blocks in threadsafe_blocks(blocksize(grid))
            Threads.@threads :static for blk in blocks
                generate!(rngs[Threads.threadid()], cells, grid, gridindices_from_blockindex(grid, blk), num_generations)
            end
        end
    else
        generate!(rng, cells, grid, CartesianIndices(size(grid)), num_generations)
    end
    filter!(!isnanvec, vec(cells))
end

# https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
function generate!(rng, cells::Array{Vec{dim, T}}, grid::Grid{dim, T}, gridindices::CartesianIndices, num_generations::Int) where {dim, T}
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
    if is_validpoint(xₖ, sampling_distance(grid), neighborcells, cells)
        # @assert isnanvec(cells[Iₖ])
        cells[Iₖ] = xₖ
        return Iₖ
    end
    nothing
end

function is_validpoint(xₖ, r, neighborcells, cells)
    valid = true
    @inbounds @simd for cellindex in neighborcells
        x = cells[cellindex]
        valid *= ifelse(isnanvec(x), true, square_sum(xₖ, x) > r^2)
    end
    valid
end

@inline square_sum(x, y) = sum(@. (x-y)^2)
@inline nanvec(::Type{Vec{dim, T}}) where {dim, T} = Vec{dim, T}(ntuple(i->NaN, Val(dim)))
@inline isnanvec(x::Vec) = x === nanvec(typeof(x))

end # module PoissonDiskSampling
