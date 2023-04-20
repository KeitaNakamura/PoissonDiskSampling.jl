using PoissonDiskSampling
using PoissonDiskSampling: Grid
using Test
using Random
using StableRNGs

@testset "misc" begin
    @testset "Grid" begin
        for T in (Float64, Float32)
            min, max = (0, 1)
            r = 0.1
            dx = r/√2
            ax = 0:dx:1
            grid = (@inferred Grid(T, r, (min,max), (min,max)))::Grid{2, T}
            @test grid.r == T(r)
            @test grid.dx == T(dx)
            @test grid.min == map(T, ntuple(i->first(ax), 2))
            @test grid.max == map(T, ntuple(i->last(ax), 2))
            @test grid.size == ntuple(i->length(ax), 2)
            @test grid.offset == zero(CartesianIndex{2})
        end
    end
    @testset "spherical coordinates" begin
        Random.seed!(1234)
        # circle
        n = 2
        r = rand()
        θs = ntuple(i->rand()*ifelse(i==n-1, π, 2π), Val(n-1))
        xs = @inferred PoissonDiskSampling.spherical_coordinates(r, θs)
        @test xs[1] ≈ r * cos(θs[1])
        @test xs[2] ≈ r * sin(θs[1])
        # sphere
        n = 3
        r = rand()
        θs = ntuple(i->rand()*ifelse(i==n-1, π, 2π), Val(n-1))
        xs = @inferred PoissonDiskSampling.spherical_coordinates(r, θs)
        @test xs[1] ≈ r * cos(θs[1])
        @test xs[2] ≈ r * sin(θs[1]) * cos(θs[2])
        @test xs[3] ≈ r * sin(θs[1]) * sin(θs[2])
        # more
        n = 5
        r = rand()
        θs = ntuple(i->rand()*ifelse(i==n-1, π, 2π), Val(n-1))
        xs = @inferred PoissonDiskSampling.spherical_coordinates(r, θs)
        @test xs[1] ≈ r * cos(θs[1])
        @test xs[2] ≈ r * sin(θs[1]) * cos(θs[2])
        @test xs[3] ≈ r * sin(θs[1]) * sin(θs[2]) * cos(θs[3])
        @test xs[4] ≈ r * sin(θs[1]) * sin(θs[2]) * sin(θs[3]) * cos(θs[4])
        @test xs[5] ≈ r * sin(θs[1]) * sin(θs[2]) * sin(θs[3]) * sin(θs[4])
    end
end

@testset "generate" begin
    # generate
    for T in (Float32, Float64)
        for parallel in (false, true)
            Random.seed!(1234)
            r = rand()
            for minmaxes in (((0,6), (-2,3)),
                             ((0,6), (-2,3), (0,2)),
                             ((0,6), (-2,3), (0,2), (-1,2)))
                n = length(minmaxes)
                dx = r / sqrt(n)
                pts = (@inferred PoissonDiskSampling.generate(T, r, minmaxes...; parallel))::Vector{NTuple{n, T}}
                # Check the distance between samples
                @test all(pts) do pt
                    all(pts) do x
                        x === pt && return true
                        sum(abs2, pt .- x) > abs2(r)
                    end
                end
                # Check if samples are uniformly distributed by calculating mean value of coordiantes.
                # The mean value should be almost the same as the centroid of the domain
                mean = collect(reduce(.+, pts)./length(pts))
                centroid = collect(map(x->(x[1]+x[2])/2, minmaxes))
                @test mean ≈ centroid atol=r
            end
            # errors
            @test_throws Exception PoissonDiskSampling.generate(T, r, (0,6); parallel)                # wrong dimension
            @test_throws Exception PoissonDiskSampling.generate(T, r, (0,6), (3,-2); parallel)        # wrong (min, max)
            @test_throws Exception PoissonDiskSampling.generate(T, r, (0,6), (-2,3), (2,0); parallel) # wrong (min, max)
        end
        # StableRNG
        rng = StableRNG(1234)
        pts = (@inferred PoissonDiskSampling.generate(rng, T, rand(rng), (0,8), (0,10); parallel=false))::Vector{NTuple{2, T}}
        centroid = collect(reduce(.+, pts) ./ length(pts))
        T == Float64 && @test centroid ≈ [3.8345015153218833, 4.83758270027716]
        T == Float32 && @test centroid ≈ [3.9360082f0, 4.9248857f0]
    end
end
