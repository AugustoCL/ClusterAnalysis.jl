# ClusterAnalysis.jl
  
```@raw html
<img src="plot_dbscan.png" width="70%">  
```

Test

This package was **BUILT FROM SCRATCH**, entirely in [Julia Lang](https://julialang.org), and implements a few popular clustering algorithms like K-Means and DBSCAN. 

This is mostly a learning experiment, but the package were also built and documented to be used by anyone, Plug-and-Play. Just input your data as an Array or a [Tables.jl](https://discourse.julialang.org/t/tables-jl-a-table-interface-for-everyone/14071) type (like [DataFrames.jl](https://dataframes.juliadata.org/stable/)), then start training your clusters algorithms and analyze your results. 

## Algorithms Implemented
Currently we implemented two types of algorithms, a partitioned based ([K-Means](https://en.wikipedia.org/wiki/K-means_clustering)) and a spatial density based ([DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)). 

> Go check the `Algorithms` Section that contains all the details of how it works the algorithm and also got the bibliography and papers used during the research and development of the code. It's a great introduction to the algorithm and a good resource to read along with the source code.

## How to install ClusterAnalysis.jl
Since this package it's not registred, it's necessary to use the github url as shown below.

```julia
# press ] to enter in Pkg REPL mode.
pkg> add https://github.com/AugustoCL/ClusterAnalysis.jl
```  
