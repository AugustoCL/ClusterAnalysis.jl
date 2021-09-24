using Statistics, LinearAlgebra
using Tables

"""
    struct KmeansResult{T<:AbstractFloat}
        K::Int
        centroids::Vector{Vector{T}}
        cluster::Vector{Int}
        withinss::T
        iter::Int
    end

Object resulting from kmeans algorithm that contains the number of clusters, centroids, clusters prediction, total-variance-within-cluster and number of iterations until convergence.
"""
struct KmeansResult{T<:AbstractFloat}
    K::Int
    centroids::Vector{Vector{T}}
    cluster::Vector{Int}
    withinss::T
    iter::Int
end


"""
    euclidean(a::AbstractVector, b::AbstractVector)

Calculate euclidean distance from two vectors. √∑(aᵢ - bᵢ)²
"""
function euclidean(a::AbstractVector{T}, 
                   b::AbstractVector{T}) where {T<:AbstractFloat}              
    @assert length(a) == length(b)

    # euclidean(a, b) = √∑(aᵢ- bᵢ)²
    s = zero(T)
    @simd for i in eachindex(a)
        @inbounds s += (a[i] - b[i])^2
    end
    return √s
end

"""
    squared_error(data::AbstractMatrix)
    squared_error(col::AbstractVector)

Function that evaluate the kmeans, using the Sum of Squared Error (SSE).
"""
function squared_error(data::AbstractMatrix{T}) where {T<:AbstractFloat}    
    error = zero(T)
    @simd for i in 1:size(data, 2)
        error += squared_error(view(data, :, i))
    end
    return error
end

function squared_error(col::AbstractVector{T}) where {T<:AbstractFloat}
    m = mean(col)
    error = zero(T)
    @simd for i in eachindex(col)
        @inbounds error += (col[i] - m)^2
    end
    return error
end


"""
    totalwithinss(data::AbstractMatrix, K::Int, cluster::Vector)

Calculate the total-variance-within-cluster using the squared_error function.
"""
function totalwithinss(data::AbstractMatrix{T}, K::Int, cluster::Vector{Int}) where {T<:AbstractFloat}
    # evaluate total-variance-within-clusters
    error = zero(T)
    @simd for k in 1:K
        error += squared_error(data[cluster .== k, :])
    end
    return error
end

"""
    kmeans(table, K::Int; nstart::Int = 10, maxiter::Int = 10)
    kmeans(data::AbstractMatrix, K::Int; nstart::Int = 10, maxiter::Int = 10)

Classify all data observations in k clusters by minimizing the total-variance-within each cluster.


Pseudo-code of the algorithm:  
* Repeat `nstart` times:  
    1. Random initialize `K` clusters centroids.  
    2. Estimate clusters.  
    3. Repeat `maxiter` times:  
        * Update centroids using the mean().  
        * Estimate clusters.  
        * Calculate the total-variance-within-cluster.  
        * Evaluate the stop rule.  
* Keep the best result of all `nstart` executions.

For more detailed explanation of the algorithm, check the [`Algorithm's Overview of KMeans`](https://github.com/AugustoCL/ClusterAnalysis.jl/blob/main/algo_overview/kmeans_overview.md)  
""" 
function kmeans(table, K::Int; nstart::Int = 10, maxiter::Int = 10)
    Tables.istable(table) ? (data = Tables.matrix(table)) : throw(ArgumentError("The table argument passed does not implement the Tables.jl interface.")) 
    return kmeans(data, K, nstart=nstart, maxiter=maxiter)
end

function kmeans(data::AbstractMatrix{T}, K::Int; nstart::Int = 10, maxiter::Int = 10) where {T}
    return kmeans(Matrix{Float64}(data), K, nstart=nstart, maxiter=maxiter)
end

function kmeans(data::AbstractMatrix{T}, K::Int; nstart::Int = 10, maxiter::Int = 10) where {T<:AbstractFloat}
    
    # generate variables to update with the best result
    nl, nc = size(data)

    centroids = [Vector{T}(undef, nc) for _ in 1:K]
    cluster = Vector{Int}(undef, nl)
    withinss = Inf
    iter = 0
    
    # run multiple kmeans to get the best result
    for _ in 1:nstart
        
        new_centroids, new_cluster, new_withinss, new_iter = _kmeans(data, K, maxiter)
        
        if new_withinss < withinss
            centroids .= new_centroids
            cluster .= new_cluster
            withinss = new_withinss
            iter = new_iter
        end
    end

    return KmeansResult(K, centroids, cluster, withinss, iter)
end

function _kmeans(data::AbstractMatrix{T}, K::Int, maxiter::Int) where {T<:AbstractFloat}
    
    # generate random centroids
    nl = size(data, 1)
    indexes = rand(1:nl, K)
    centroids = Vector{T}[data[i, :] for i in indexes]

    # first clusters estimate
    cluster = Vector{Int}(undef, nl)
    for (i, obs) in enumerate(eachrow(data))
        dist = [euclidean(obs, c) for c in centroids]
        @inbounds cluster[i] = argmin(dist)
    end

    # first evaluation of total-variance-within-cluster
    withinss = totalwithinss(data, K, cluster)

    # variables to update during the iterations
    new_centroids = copy(centroids)
    new_cluster = copy(cluster)
    iter = 1
    norms = norm.(centroids)
    
    # start kmeans iterations until maxiter or convergence
    for _ in 2:maxiter

        # update new_centroids using the mean
        @simd for k in 1:K             # mean.(eachcol(data[new_cluster .== k, :]))
            @inbounds new_centroids[k] = vec(mean(view(data, new_cluster .== k, :), dims = 1))
        end

        # estimate cluster to all observations
        for (i, obs) in enumerate(eachrow(data))
            dist = [euclidean(obs, c) for c in new_centroids]
            @inbounds new_cluster[i] = argmin(dist)
        end

        # update iter, withinss-variance and calculate centroid norms
        new_withinss = totalwithinss(data, K, new_cluster)       
        new_norms = norm.(new_centroids)
        iter += 1
        
        # convergence rule
        if norm(norms - new_norms) ≈ 0
            break
        end

        # update centroid norms
        norms .= new_norms

        # update centroids, cluster and whithinss
        if new_withinss < withinss
            centroids .= new_centroids
            cluster .= new_cluster
            withinss = new_withinss
        end

    end

    return centroids, cluster, withinss, iter
end