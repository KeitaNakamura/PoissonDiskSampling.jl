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
pkg> add https://github.com/KeitaNakamura/PoissonDiskSampling.jl.git
```

## How to use

See `?PoissonDiskSampling.generate` for details.

```julia
julia> using PoissonDiskSampling, Plots

julia> points = PoissonDiskSampling.generate((0,5), (0,3); r = 0.1);

julia> scatter(points)
```

![demo](https://github.com/KeitaNakamura/PoissonDiskSampling.jl/blob/main/assets/demo.svg)
