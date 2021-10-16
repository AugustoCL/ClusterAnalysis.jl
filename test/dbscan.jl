@testset "DBSCAN" begin

    Random.seed!(42)

    df = CSV.read("data/blob_data.csv", DataFrame, drop=[1]);

    X = df[:,1:2];
    ϵ = 0.35;
    min_pts = 10;
    
    X_int = round.(Int, X.*100)
    ϵ_int = 35.0;
    
    @testset "test dimensions results" begin 
        res = dbscan(X, ϵ, min_pts)
        n_clusters = unique(res.labels) |> length
        @test length(res.labels) == size(X, 1)
        @test length(res.clusters) == n_clusters - 1
        @test length(res.clusters[1]) == 1010
        @test length(res.clusters[2]) == 999
        @test length(res.clusters[3]) == 994
        @test length(res.clusters[4]) == 1000
    end

    @testset "test Integer conversion" begin
        res_int = dbscan(X_int,ϵ_int, min_pts)
        @test eltype(res_int.df) == Float64
        @test length(res_int.labels) == size(X_int, 1)
    end

    @testset "test errors" begin
        @test_throws ArgumentError dbscan(X[:, 2], ϵ, min_pts)
    end
end
