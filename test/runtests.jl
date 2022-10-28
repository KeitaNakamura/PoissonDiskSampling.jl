using PoissonDiskSampling
using Test

@testset "PoissonDiskSampling" begin
    r = rand()
    for points in (PoissonDiskSampling.generate(r, (0,8), (-3,4)),
                   PoissonDiskSampling.generate(r, (0,10), (-3,4), (0,8), ),)
        @test all(points) do pt
            all(points) do x
                x === pt && return true
                sum(abs2, pt .- x) > abs2(r)
            end
        end
    end
end
