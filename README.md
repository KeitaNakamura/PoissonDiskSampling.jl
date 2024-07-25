<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/KeitaNakamura/PoissonDiskSampling.jl/blob/main/assets/logo-light.png">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/KeitaNakamura/PoissonDiskSampling.jl/blob/main/assets/logo-dark.png">
  <img alt="logo" src="https://github.com/KeitaNakamura/PoissonDiskSampling.jl/blob/main/assets/logo-light.png" width=600>
</picture>

*n-dimensional Poisson disk sampling for Julia*

[![Build Status](https://github.com/KeitaNakamura/PoissonDiskSampling.jl/workflows/CI/badge.svg)](https://github.com/KeitaNakamura/PoissonDiskSampling.jl/actions)
[![codecov](https://codecov.io/gh/KeitaNakamura/PoissonDiskSampling.jl/branch/main/graph/badge.svg?token=7vrwuWCsYU)](https://codecov.io/gh/KeitaNakamura/PoissonDiskSampling.jl)

## Installation

```julia
pkg> add PoissonDiskSampling
```

## How to use

See `?PoissonDiskSampling.generate` for details.

```julia
julia> using PoissonDiskSampling

julia> r = 0.1 # minimum distance between samples

julia> points = PoissonDiskSampling.generate(r, (0,5), (0,3));

julia> typeof(points)
Vector{Tuple{Float64, Float64}} (alias for Array{Tuple{Float64, Float64}, 1})

julia> using Plots

julia> scatter(points)
```

![demo](https://github.com/KeitaNakamura/PoissonDiskSampling.jl/blob/main/assets/demo.svg)
