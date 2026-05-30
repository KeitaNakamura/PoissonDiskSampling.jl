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
            part = PoissonDiskSampling.partition(grid, CartesianIndices((3:7, 5:9)))
            @test part.r == grid.r
            @test part.dx == grid.dx
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
    @testset "annulus" begin
        for T in (Float64, Float32), n in (1, 2, 3, 5)
            rng = StableRNG(1234)
            annulus = PoissonDiskSampling.Annulus(ntuple(_ -> zero(T), Val(n)), T(1), T(2))
            x = (@inferred PoissonDiskSampling.random_point(rng, annulus))::NTuple{n, T}
            d² = sum(x .^ 2)
            @test T(1)^2 ≤ d² ≤ T(2)^2
        end

        rng = StableRNG(1234)
        n = 3
        annulus = PoissonDiskSampling.Annulus((0.0, 0.0, 0.0), 1.0, 2.0)
        samples = [PoissonDiskSampling.random_point(rng, annulus) for _ in 1:3000]
        radial = map(samples) do x
            d = sqrt(sum(x .^ 2))
            (d^n - annulus.r1^n) / (annulus.r2^n - annulus.r1^n)
        end
        direction = map(samples) do x
            x[1]^2 / sum(x .^ 2)
        end
        @test sum(radial) / length(radial) ≈ 0.5 atol=0.03
        @test sum(direction) / length(direction) ≈ 1/n atol=0.03
    end
end

@testset "generate" begin
    # generate
    for T in (Float32, Float64)
        for multithreading in (false, true)
            Random.seed!(1234)
            r = rand(T)
            for minmaxes in (((0,6),),
                             ((0,6), (-2,3)),
                             ((0,6), (-2,3), (0,2)),
                             ((0,6), (-2,3), (0,2), (-1,2)))
                n = length(minmaxes)
                dx = r / sqrt(n)
                pts = (@inferred PoissonDiskSampling.generate(T, r, minmaxes...; multithreading))::Vector{NTuple{n, T}}
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
            @test_throws ArgumentError PoissonDiskSampling.generate(T, zero(T), (0,6), (-2,3); multithreading)
            @test_throws ArgumentError PoissonDiskSampling.generate(T, -r, (0,6), (-2,3); multithreading)
            @test_throws ArgumentError PoissonDiskSampling.generate(T, T(Inf), (0,6), (-2,3); multithreading)
            @test_throws ArgumentError PoissonDiskSampling.generate(T, T(NaN), (0,6), (-2,3); multithreading)
            @test_throws ArgumentError PoissonDiskSampling.generate(T, r, (0,6), (-2,3); k=0, multithreading)
            @test_throws ArgumentError PoissonDiskSampling.generate(T, r, (0,6), (-2,3); k=-1, multithreading)
            @test_throws ArgumentError PoissonDiskSampling.generate(T, r; multithreading)                      # wrong dimension
            @test_throws ArgumentError PoissonDiskSampling.generate(T, r, (0,6), (3,-2); multithreading)        # wrong (min, max)
            @test_throws ArgumentError PoissonDiskSampling.generate(T, r, (0,6), (-2,3), (2,0); multithreading) # wrong (min, max)
        end
        # StableRNG
        rng = StableRNG(1234)
        pts = (@inferred PoissonDiskSampling.generate(rng, T, rand(rng), (0,8), (0,10)))::Vector{NTuple{2, T}}
        centroid = collect(reduce(.+, pts) ./ length(pts))
        T == Float64 && @test centroid ≈ [4.028021944848475, 5.112373302283751]
        T == Float32 && @test centroid ≈ [3.932441f0, 5.0294213f0]
    end

    if Threads.nthreads() > 1
        @testset "multithreaded reproducibility" begin
            for T in (Float32, Float64), RNG in (MersenneTwister, StableRNG)
                r = T(0.1)
                pts1 = PoissonDiskSampling.generate(RNG(1234), T, r, (0,5), (0,3); multithreading=true)
                pts2 = PoissonDiskSampling.generate(RNG(1234), T, r, (0,5), (0,3); multithreading=true)
                @test pts1 == pts2
            end
        end
    end
end
