## 1. Types declaration
abstract type Silhouette end
struct Common <: Silhouette end
struct Simplified <: Silhouette end

## 2. Functions
"""

    euclidean(a::AbstractVector, b::AbstractArray)

Calculate the euclidean distance between a data point and bunch of data points.

# Arguments (positional)
- `a`: First vector.
- `b`: Second vector.

# Returns
A Float64 value as the distance.

"""
function euclidean(a::AbstractVector, b::AbstractArray)
    √(sum((a' .- b).^2))
end

"""

    aᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Common) where {T<:Real, S<:Real}
    aᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Simplified, centers) where {T<:Real, S<:Real}

Calculate the euclidean distance between i'th sample and data points in the same
cluster.

# Arguments (positional)
- `data`: data set.
- `labels`: cluster identity of each sample
- `i`: The index of the sample.

# Returns
A Float64 value that represents sum distance between sample i and data points
in the same cluster.
"""

function aᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Common) where {T<:Real, S<:Real}
    labelᵢ = labels[i]
    same_cluster_members_idx = findall(isequal(labelᵢ), labels)
    n = length(same_cluster_members_idx)

    sum_dist = euclidean(data[i, :], data[same_cluster_members_idx, :])
    return sum_dist/(n-1)
end

function aᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Simplified, centers) where {T<:Real, S<:Real}
    labelᵢ = labels[i]
    return euclidean(data[i, :], centers[labelᵢ]')
end

"""
    bᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Common, centers) where {T<:Real, S<:Real}
    bᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Simplified, centers) where {T<:Real, S<:Real}

Calculate the euclidean distance between i'th sample and data points in other clusters.

# Arguments (positional)
- `data`: data set.
- `labels`: cluster identity of each sample
- `i`: The index of the sample.
- `method`: Common or Simplified version of silhouette.
- `centers`: Cluster centers.

# Returns
A Float64 value that represents sum distance between sample i and data points in other clusters.
"""

function bᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Common, centers) where {T<:Real, S<:Real}
    labelᵢ = labels[i]
    dissim_labels = [idx for idx=1:length(centers) if idx!=labelᵢ]
    mean_dist = similar(dissim_labels, Float64)
    idx = 0
    for (idx,j) in enumerate(dissim_labels)
        related_idx = findall(isequal(j), labels)
        @inbounds mean_dist[idx] = euclidean(data[i, :], data[related_idx, :])
    end

    return minimum(mean_dist)
end

function bᵢ(data::AbstractArray{T}, labels::AbstractVector{S}, i::Int64, method::Simplified, centers) where {T<:Real, S<:Real}
    labelᵢ = labels[i]
    clusters_to_iterate = [idx for idx=1:length(centers) if idx!=labelᵢ]
    center = vcat(transpose.(centers)...)
    mean_dist = [euclidean(data[i, :], center[clus_idx, :]) for clus_idx in clusters_to_iterate]

    return minimum(mean_dist)
end

"""
    sᵢ(aᵢ, bᵢ)

Calculate the silhouette of i'th sample.

# Arguments (positional)
- `aᵢ`: The sum distance between i'th sample and data points in the same cluster.
- `bᵢ`: The sum distance between i'th sample and data points in other clusters.

# Returns
A Float64 value that represents silhouette of i'th sample.

"""
function sᵢ(aᵢ, bᵢ)
    return (bᵢ - aᵢ)/max(aᵢ, bᵢ)
end

"""
    Silouhette(input_data, model::KmeansResult)

Calculate the silhouette coefficient of the clustering result.

# Arguments (positional)
- `input_data`: data set.
- `model`: an `KmeansResult` object.

# Returns
A Float64 value that represents silhouette coefficient of the clustering result.
"""
function Silouhette(input_data, model::KmeansResult)
    Tables.istable(input_data) ? data=Tables.matrix(input_data) : error()
    labels = model.cluster
    centers = model.centroids
    if big(size(data, 1))^2 > 10^3
        method = Simplified()
        sᵢs = [
            sᵢ(
                aᵢ(data, labels, i, method, centers),
                bᵢ(data, labels, i, method, centers)
            )
            for i=1:size(data, 1)
        ]
    else
        method = Common()
        sᵢs = [
            sᵢ(
                aᵢ(data, labels, i, method),
                bᵢ(data, labels, i, method, centers)
            )
            for i=1:size(data, 1)
        ]
    end

    return mean(sᵢs)
end
