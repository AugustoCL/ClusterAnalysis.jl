using Test, ClusterAnalysis
using CSV, DataFrames
using Random

@testset "ClusterAnalysis" begin

include("kmeans.jl");
include("dbscan.jl");

end
