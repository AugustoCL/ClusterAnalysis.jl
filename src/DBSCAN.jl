"""
    struct DBSCAN{T<:AbstractFloat, KD<:KDTree}
        df::Matrix{T}
        ϵ::T
        min_pts::Int
        labels::Vector{Int}
        tree::KD
        clusters::Vector{Vector{Int}}
    end

Struct that contains all the relevant information about the model. The `data`, `ϵ`, `min_pts`, `labels`, `KDTree` and `clusters`.
The labels and the clusters are defined during the algorithm inside the routine inside `fit!()` function, which iterate over every observation p in the dataset.

Internal/External Constructors  

* Int - DBSCAN(df::Matrix{T}, ϵ::T, min_pts::Int) where {T<:AbstractFloat}   
* Ext - DBSCAN(table, ϵ::Real, min_pts::Int)  
* Ext - DBSCAN(df::Matrix{T}, ϵ::Real, min_pts::Int) where {T<:AbstractFloat} = DBSCAN(df, T(ϵ), min_pts)  
* Ext - DBSCAN(df::Matrix{T}, ϵ::Real, min_pts::Int) where {T} = DBSCAN(Matrix{Float64}(df), ϵ, min_pts)  
"""
struct DBSCAN{T<:AbstractFloat, KD<:KDTree}
    df::Matrix{T}
    ϵ::T
    min_pts::Int
    labels::Vector{Int}
    tree::KD
    clusters::Vector{Vector{Int}}

    function DBSCAN(df::Matrix{T}, ϵ::T, min_pts::Int) where {T<:AbstractFloat} 
        labels = fill(-1, size(df,1))
        tree = KDTree(df', leafsize=20)
        clusters = Vector{Vector{Int}}()
        KD = typeof(tree)

        new{T, KD}(df, ϵ, min_pts, labels, tree, clusters)
    end
end

# External Constructors
DBSCAN(df::Matrix{T}, ϵ::Real, min_pts::Int) where {T<:AbstractFloat} = DBSCAN(df, T(ϵ), min_pts)
DBSCAN(df::Matrix{T}, ϵ::Real, min_pts::Int) where {T} = DBSCAN(Matrix{Float64}(df), ϵ, min_pts)

function DBSCAN(table, ϵ::Real, min_pts::Int)
    Tables.istable(table) ? (df = Tables.matrix(table)) : throw(ArgumentError("The df argument passed does not implement the Tables.jl interface."))
    return DBSCAN(df, ϵ, min_pts)
end


"""
    dbscan(df, ϵ::Real, min_pts::Int)

Classify data observations in clusters and noises by using a density concept obtained with the parameters input (`ϵ`, `min_pts`).  

The number of clusters are obtained during the execution of the model, therefore, initially the user don't know how much clusters it will obtain.  
The algorithm use the `KDTree` structure from `NearestNeighbors.jl` to calculate the `RangeQuery` operation more efficiently.  

For more detailed explanation of the algorithm, check the [`Algorithm's Overview of DBSCAN`](https://github.com/AugustoCL/ClusterAnalysis.jl/blob/main/algo_overview/dbscan_overview.md)  
""" 
function dbscan(df, ϵ::Real, min_pts::Int)
    model = DBSCAN(df, ϵ, min_pts)
    fit!(model)
    return model
end


function fit!(model::DBSCAN)
    nlines = size(model.df, 1)                  # Number of observations/points in dataset
    visited = falses(nlines)                    # Array to control if point was visited
    clusters_selected = falses(nlines)          # Array to identify if point was labeled in for-loop
    FIFO = Int[]                                # FIFO data structure
    nb_list = Int[]                             # Array with the neighbors's point

    # for each point p in data
    for p ∈ 1:nlines                      
        visited[p] && continue                  # Check if point p was visited
        push!(FIFO, p)                          # Add p in FIFO (First In First Out) to execute RangeQuery
        fill!(clusters_selected, false)
       
        # start RangeQuery all points in FIFO until finished
        # during loop the FIFO got updated with more points
        while !isempty(FIFO)
            newp = popfirst!(FIFO)              # Remove and return the first element in FIFO
            visited[newp] && continue           # Check if point `newp` was visited
            visited[newp] = true                # assign p as visited
            
            nbs = inrange(model.tree,           # get all the neighbors (index) from point `newp` 
                          model.df[newp, :],    # it's the RangeQuery from NearestNeighbors.jl
                          model.ϵ)
        
            # later, try remove nb_list
            append!(nb_list, nbs)               # save neighbor's indexes in nb_list
            clusters_selected[nb_list] .= true 

            if length(nb_list) < model.min_pts  # check if point is core
                empty!(nb_list)
                continue
            end

            for i in nb_list                    # update the FIFO wih nbs
                visited[i] && continue
                push!(FIFO, i)
            end

            empty!(nb_list)                     # clean list to evaluate new point
        end
        
        # save the indices from each cluster 
        ind_cluster = findall(clusters_selected)    
        model.min_pts ≤ sum(clusters_selected) && push!(model.clusters, ind_cluster)
    end

    # write labels to struct and disregards duplicate indices    
    nc = 1
    for cl in model.clusters
        for i in cl
            model.labels[i] = nc
        end
        nc += 1
    end
end