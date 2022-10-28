using PoissonDiskSampling
using Test

@testset "PoissonDiskSampling" begin
    # generate
    r = rand()
    for points in (PoissonDiskSampling.generate(r, (0,8), (-3,4)),
                   PoissonDiskSampling.generate(r, (0,10), (-3,4), (0,8)),)
        @test all(points) do pt
            all(points) do x
                x === pt && return true
                sum(abs2, pt .- x) > abs2(r)
            end
        end
    end
    # errors
    @test_throws ArgumentError PoissonDiskSampling.generate(r, (8,0), (-3,4))
    @test_throws ArgumentError PoissonDiskSampling.generate(r, (0,10), (4,-3), (0,8))
end
