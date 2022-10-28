module PoissonDiskSampling

const Vec{dim, T} = NTuple{dim, T}

########
# Grid #
########

struct Grid{dim}
    dx::Float64
    min::NTuple{dim, Float64}
    max::NTuple{dim, Float64}
    size::NTuple{dim, Int}
end
Base.size(grid::Grid) = grid.size

function Grid(dx::Real, minmaxes::Vararg{Tuple{Real, Real}, dim}) where {dim}
    all(minmax->minmax[1]<minmax[2], minmaxes) || throw(ArgumentError("`(min, max)` must be `min < max`"))
    axes = map(minmax->range(minmax...; step=dx), minmaxes)
    min = map(first, axes)
    max = map(last, axes)
    Grid{dim}(dx, min, max, map(length, axes))
end

function whichcell(grid::Grid, x::Vec{dim, T}) where {dim, T}
    xmin = grid.min
    dx⁻¹ = inv(grid.dx)
    ξ = @. (x - xmin) * dx⁻¹
    ncells = size(grid) .- 1
    all(@. ξ == ncells) && return CartesianIndex(ncells)
    all(@. 0 ≤ ξ < ncells) || return nothing # use `<` because of `floor`
    CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

function random_point(grid::Grid)
    map(grid.min, grid.max) do xmin, xmax
        xmin + rand() * (xmax - xmin)
    end
end

###########
# Annulus #
###########

struct Annulus{dim}
    centroid::Vec{dim, Float64}
    r1::Float64
    r2::Float64
end

function random_point(annulus::Annulus{1})
    r = annulus.r1 + rand() * (annulus.r2 - annulus.r1)
    sign = ifelse(rand(Bool), 1, -1)
    x = sign * r
    annulus.centroid .+ x
end

function random_point(annulus::Annulus{2})
    r = annulus.r1 + rand() * (annulus.r2 - annulus.r1)
    θ = rand() * 2π
    x = r * cos(θ)
    y = r * sin(θ)
    annulus.centroid .+ (x, y)
end

function random_point(annulus::Annulus{3})
    r = annulus.r1 + rand() * (annulus.r2 - annulus.r1)
    θ = rand() * π
    ϕ = rand() * 2π
    x = r * sin(θ) * cos(ϕ)
    y = r * sin(θ) * sin(ϕ)
    z = r * cos(θ)
    annulus.centroid .+ (x, y, z)
end

#########################
# Poisson disk sampling #
#########################

function generate(r::Real, minmaxes::Vararg{Tuple{Real, Real}, dim}; num_generations::Int = 30) where {dim}
    dx = r / √dim
    grid = Grid(dx, minmaxes...)
    cells = fill(-1, size(grid).-1)

    active_list = Int[]
    points = Vec{dim, Float64}[]

    push!(points, random_point(grid))
    push!(active_list, length(points))
    cells[whichcell(grid, points[end])] = length(points)

    while !isempty(active_list)
        index = rand(1:length(active_list))
        x = points[active_list[index]]
        found = false
        for k in 1:num_generations
            x_k = random_point(Annulus(x, r, 2r))
            I = whichcell(grid, x_k)
            I === nothing && continue
            u = 2*oneunit(I)
            neighborcells = CartesianIndices(cells) ∩ ((I-u):(I+u))
            valid = all(neighborcells) do cellid
                ptid = cells[cellid]
                ptid == -1 && return true
                p = points[ptid]
                sum(abs2, x_k .- p) > abs2(r)
            end
            if valid
                @assert cells[I] == -1
                push!(points, x_k)
                push!(active_list, length(points))
                cells[I] = length(points)
                found = true
            end
        end
        !found && popat!(active_list, index)
    end

    points
end

end # module PoissonDiskSampling
