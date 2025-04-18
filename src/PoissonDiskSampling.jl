module PoissonDiskSampling

using Random

const BLOCKFACTOR = unsigned(3) # 2^3
const Vec{dim, T} = NTuple{dim, T}

"""
    Grid(r, (min_1, max_1)..., (min_n, max_n))

Construct a grid with the domain ``[min_1, max_1)`` ... ``[min_n, max_n)``,
where ``max_n`` may be changed based on the minimum distance `r` between samples.
"""
struct Grid{dim, T}
    r::T
    dx::T
    xmin::NTuple{dim, T}
    xmax::NTuple{dim, T}
    dims::NTuple{dim, Int}
end
Base.size(grid::Grid) = grid.dims
@inline sampling_distance(grid::Grid) = grid.r

function Grid(::Type{T}, r::Real, minmax::Vararg{Tuple{Real, Real}, n}) where {T, n}
    all(Base.splat(<), minmax) || throw(ArgumentError("`(min, max)` must be `min < max`"))
    dx = r/√n
    xmin = getindex.(minmax, 1)
    xmax = getindex.(minmax, 2)
    dims = @. ceil(Int, (xmax-xmin)/dx) + 1
    Grid{n, T}(r, dx, xmin, xmax, dims)
end

function whichcell(x::Vec{dim, T}, grid::Grid{dim, T}) where {dim, T}
    xmin, xmax = grid.xmin, grid.xmax
    all(@. xmin ≤ x < xmax) || return nothing
    dx⁻¹ = inv(grid.dx)
    ξ = @. (x - xmin) * dx⁻¹
    CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

function neighborcells(x::Vec{dim, T}, grid::Grid{dim, T}) where {dim, T}
    xmin = grid.xmin
    dx⁻¹ = inv(grid.dx)
    ξ = @. (x - xmin) * dx⁻¹
    r = grid.r * dx⁻¹
    start = CartesianIndex(@. unsafe_trunc(Int, floor(ξ - r)) + 1)
    stop  = CartesianIndex(@. unsafe_trunc(Int, floor(ξ + r)) + 1)
    CartesianIndices(size(grid).-1) ∩ (start:stop)
end

function random_point(rng, grid::Grid{dim, T}) where {dim, T}
    map(grid.xmin, grid.xmax) do xmin, xmax
        xmin + rand(rng, T) * (xmax - xmin)
    end
end

function partition(grid::Grid{dim}, CI::CartesianIndices{dim}) where {dim}
    @boundscheck checkbounds(CartesianIndices(size(grid)), CI)
    Imin = first(CI).I
    Imax = last(CI).I
    new_min = @. grid.xmin + grid.dx * (Imin - 1)
    new_max = @. grid.xmin + grid.dx * (Imax - 1)
    Grid(grid.dx, grid.r, new_min, new_max, size(CI))
end

# block methods
blocksize(gridsize::Tuple{Vararg{Int}}) = @. (gridsize-1)>>BLOCKFACTOR+1
blocksize(grid::Grid) = blocksize(size(grid))
function threadsafe_blocks(blocksize::NTuple{dim, Int}) where {dim}
    starts = collect(Iterators.product(ntuple(i->1:2, Val(dim))...))
    vec(map(st -> map(CartesianIndex{dim}, Iterators.product(StepRange.(st, 2, blocksize)...)), starts))
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
    PoissonDiskSampling.generate([rng=GLOBAL_RNG], [T=Float64], r, (min_1, max_1)..., (min_n, max_n); k=30, multithreading=false)

Geneate points based on Poisson disk sampling.

The domain must be rectangular, defined as ``[min_1, max_1)`` ... ``[min_n, max_n)``.
`r` is the minimum distance between samples. `k` is the number of trials for sampling,
i.e., the algorithm will give up if no valid sample is found after `k` trials.

The algorithm is based on *https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf*.
"""
generate(args...; k::Int=30, multithreading::Bool=false) = _generate(args...; k, multithreading)
_generate(rng, ::Type{T}, r, minmaxes::Tuple{Real, Real}...; k, multithreading) where {T} = generate(rng, Grid(T, r, minmaxes...), k, multithreading)
_generate(rng,            r, minmaxes::Tuple{Real, Real}...; k, multithreading)           = _generate(rng, Float64, r, minmaxes...; k, multithreading)
_generate(     ::Type{T}, r, minmaxes::Tuple{Real, Real}...; k, multithreading) where {T} = _generate(Random.default_rng(), T, r, minmaxes...; k, multithreading)
_generate(                r, minmaxes::Tuple{Real, Real}...; k, multithreading)           = _generate(Random.default_rng(), r, minmaxes...; k, multithreading)

function generate(rng, grid::Grid{dim, T}, num_generations::Int, multithreading::Bool) where {dim, T}
    cells = fill(nanvec(Vec{dim, T}), size(grid).-1)
    if multithreading && Threads.nthreads() > 1
        for blocks in threadsafe_blocks(blocksize(grid))
            Threads.@threads for blk in blocks
                generate!(rng, cells, grid, gridindices_from_blockindex(grid, blk), num_generations)
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
        for k in 1:num_generations
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
    if is_validpoint(xₖ, sampling_distance(grid), neighborcells(xₖ, grid), cells)
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
        valid *= !(square_sum(xₖ.-x) ≤ r^2)
    end
    valid
end

@inline square_sum(x::Vec{2}) = muladd(x[1], x[1], x[2]*x[2])
@inline square_sum(x::Vec{3}) = muladd(x[1], x[1], muladd(x[2], x[2], x[3]*x[3]))
@inline square_sum(x) = sum(x.^2)
@inline nanvec(::Type{Vec{dim, T}}) where {dim, T} = Vec{dim, T}(ntuple(i->NaN, Val(dim)))
@inline isnanvec(x::Vec) = x === nanvec(typeof(x))

end # module PoissonDiskSampling
