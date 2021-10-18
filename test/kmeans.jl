@testset "K-Means" begin
    
    Random.seed!(42)

    @testset "Internal functions" begin
        # data inputs
        a = Float64[1, 3, 5, 7, 11, 13]
        b = Float64[2, 4, 6, 8, 12, 14]
        A = Float64[1 3 5; 7 11 13]
        B = Float64[1 3; 5 7; 11 13]
        
        # euclidean
        @test ClusterAnalysis.euclidean(a, b) == 2.449489742783178
    
        # squared_error in vector and matrix
        @test ClusterAnalysis.squared_error(a) == 107.33333333333334
        @test ClusterAnalysis.squared_error(A[1, :]) == 8.0

        # totalwithinss
        cluster = [1, 1, 2]
        @test ClusterAnalysis.totalwithinss(B, 2, cluster) == 16.0
    end

    @testset "Kmeans Results" begin
        A = rand(100, 4)
        k = 4
            
        @testset "test dimensions results" begin 
            res1 = kmeans(A, k, init=:kmpp)
            @test length(res1.centroids) == k
            @test length(res1.cluster) == size(A, 1)
            @test length(res1.centroids[1]) == size(A, 2)
        end

        @testset "diferent types of float" begin
            B = rand(Float32, 100, 4)
            res2 = kmeans(B, k, init=:random)
            @test eltype(B) == eltype(res2.centroids[1])
        end
        
        @testset "convert Int to float" begin
            C = rand(Int64, 100, 4)
            res3 = kmeans(C, k)
            @test Float64 == eltype(res3.centroids[1])
        end

        @testset "test errors" begin
            @test_throws ArgumentError kmeans(A, k, init=:wrongargument)
            @test_throws ArgumentError kmeans([1, 4, 5, 6, 7], k)
            @test_throws MethodError kmeans(A, 1.2)
        end
    end
end
