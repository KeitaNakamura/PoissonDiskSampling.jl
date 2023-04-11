using PoissonDiskSampling
using Test
using Random
using StableRNGs

@testset "Misc" begin
    @testset "Spherical coordinates" begin
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

@testset "PoissonDiskSampling" begin
    # generate
    for parallel in (false, true)
        Random.seed!(1234)
        r = rand()
        for minmaxes in (((0,6), (-2,3)),
                         ((0,6), (-2,3), (0,2)),
                         ((0,6), (-2,3), (0,2), (-1,2)))
            dx = r / sqrt(length(minmaxes))
            pts = PoissonDiskSampling.generate(r, minmaxes...; parallel)
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
        @test_throws Exception PoissonDiskSampling.generate(r, (0,6); parallel)                # wrong dimension
        @test_throws Exception PoissonDiskSampling.generate(r, (0,6), (3,-2); parallel)        # wrong (min, max)
        @test_throws Exception PoissonDiskSampling.generate(r, (0,6), (-2,3), (2,0); parallel) # wrong (min, max)
    end
    # StableRNG
    rng = StableRNG(1234)
    pts = PoissonDiskSampling.generate(rng, rand(rng), (0,8), (0,10); parallel=false)
    @test collect(reduce(.+, pts) ./ length(pts)) ≈ [3.8345015153218833, 4.83758270027716]
end
