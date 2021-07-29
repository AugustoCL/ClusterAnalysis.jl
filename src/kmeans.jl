using Statistics
using Tables

"""
    d_eucl(L1::AbstractArray, L2::AbstractArray)

Calculate euclidian distance from two arrays.
"""
function d_eucl(L1::AbstractArray, L2::AbstractArray)
    @assert length(L1) == length(L2)
    dist = sum( (L1 - L2).^2 )
    return sqrt(dist)
end

"""
    Kmeans(df::Matrix{T}, K::Int) where {T<:AbstractFloat}

Create the K-means cluster model and also initialize calculating the first centroids, estimating clusters and total variance.

# Constructors
```julia 
Kmeans(df::Matrix{T}, K::Int) where {T} = Kmeans(Matrix{Float64}(df), K)
Kmeans(df::DataFrame, K::Int) = Kmeans(Matrix{Float64}(df), K)
```

# Fields
- df::Matrix{T}: return the dataset in a Matrix form.
- K::Int: return the number of cluster of the model.
- centroids::Vector{Vector{T}}: returns the values of each variable for each centroid 
- cluster::Vector{T}: return the cluster output for each observation in the dataset.
- variance::T: return the total variance of model, summing the variance of each cluster.

"""
mutable struct Kmeans{T<:AbstractFloat}
    df::Matrix{T}
    K::Int
    centroids::Vector{Vector{T}}
    cluster::Vector{T}
    variance::T

    # initialization function
    function Kmeans(df::Matrix{T}, K::Int) where {T<:AbstractFloat}
        
        # generate random centroids
        nl = size(df, 1)
        indexes = rand(1:nl, K)
        centroids = Vector{T}[df[i,:] for i in indexes]

        # estimate clusters
        cluster = T[]
        for obs in eachrow(df)
            dist = [d_eucl(obs, c) for c in centroids]
            cl = argmin(dist)
            push!(cluster, cl)
        end

        # evaluate total variance
        cl = sort(unique(cluster))
        variance = zero(T)
        for k in cl
            df_filter = df[cluster .== k, :]
            variance += sum( var.( eachcol(df_filter) ) )
        end

        return new{T}(df, K, centroids, cluster, variance)
    end
end

Kmeans(df::Matrix{T}, K::Int) where {T} = Kmeans(Matrix{Float64}(df), K)

function Kmeans(df, K::Int)
    Tables.istable(df) ? (df = Tables.matrix(df)) : throw(ArgumentError("The df argument passed does not implement the Tables.jl interface.")) 
    return Kmeans(df, K)
end

"""
    iteration!(model::Kmeans{T}, niter::Int) where {T<:AbstractFloat}

Random initialize K cluster centroids, then estimate cluster and update centroid `niter` times,
calculate the total variance and evaluate if it's the optimum result.
"""
function iteration!(model::Kmeans{T}, niter::Int) where {T<:AbstractFloat}
    
    # Randomly initialize K cluster centroids
    nl = size(model.df, 1)
    indexes = rand(1:nl, model.K)
    centroids = Vector{T}[model.df[i,:] for i in indexes]
    cluster = T[]

    # estimate cluster and update centroids
    for _ in 1:niter

        # estimate cluster to all observations 
        cls = T[]
        for obs in eachrow(model.df)
            dist = [d_eucl(obs, c) for c in centroids]
            cl = argmin(dist)
            push!(cls, cl)
        end
    
        # update centroids using the mean
        cl = sort(unique(cls))
        ctr = Vector{Float64}[]
        for k in cl
            df_filter = model.df[cls .== k, :]
            push!( ctr, mean.(eachcol(df_filter)) )
        end
    
        # update the variables
        cluster = cls
        centroids = ctr
    end

    # evaluate total variance of all cluster
    cl = sort(unique(cluster))
    variance = zero(T)
    for k in cl
        df_filter = model.df[cluster .== k, :]
        variance += sum( var.(eachcol(df_filter)) )
    end

    # evaluate if update or not the kmeans model (minimizing the total variance) 
    if variance < model.variance
        model.centroids = centroids
        model.cluster = cluster
        model.variance = variance
    end
end

"""
    fit!(model::Kmeans, nstart::Int=50, niter::Int=10)

execute `nstart` times the iteration! function to try obtain the global optimum.
"""
function fit!(model::Kmeans, nstart::Int=50, niter::Int=10)
    for _ in 1:nstart
        iteration!(model, niter)
    end  
end
