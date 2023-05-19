using PoissonDiskSampling
using PoissonDiskSampling: Grid
using Test
using Random
using StableRNGs

@testset "misc" begin
    @testset "Grid" begin
        for T in (Float64, Float32)
            xmin, xmax = (0, 1)
            ymin, ymax = (0, 2)
            r = 0.1
            dx = r/√2
            grid = (@inferred Grid(T, r, (xmin,xmax), (ymin,ymax)))::Grid{2, T}
            @test grid.r == T(r)
            @test grid.dx == T(dx)
            @test grid.xmin == map(T, (xmin, ymin))
            @test grid.xmax == map(T, (xmax, ymax))
            @test grid.dims == (16,30)
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
            r = rand(T)
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
                        sum(@. (pt.-x)^2) > r^2
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
        pts = (@inferred PoissonDiskSampling.generate(rng, T, rand(rng), (0,8), (0,10)))::Vector{NTuple{2, T}}
        centroid = collect(reduce(.+, pts) ./ length(pts))
        T == Float64 && @test centroid ≈ [4.0275421751736395, 4.957609421464297]
        T == Float32 && @test centroid ≈ [4.090827f0, 5.0170517f0]
    end
end
