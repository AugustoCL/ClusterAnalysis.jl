using NearestNeighbors, Tables

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


# Algorithm DBSCAN logic 
function dbscan(df, ϵ::Real, min_pts::Int)
    model = DBSCAN(df, ϵ, min_pts)
    fit!(model)
    return model
end

function fit!(model::DBSCAN)
    nlines = size(model.df, 1)
    visited = falses(nlines)
    clusters_selected = falses(nlines)
    FIFO = Vector{Int}()
    nb_list = Vector{Int}()

    for p ∈ 1:nlines                      
        visited[p] && continue
        push!(FIFO, p)
        fill!(clusters_selected, false)
       
        while !isempty(FIFO)
            newp = popfirst!(FIFO)
            visited[newp] && continue
            visited[newp] = true
            nbs = inrange(model.tree, model.df[newp, :], model.ϵ)
            append!(nb_list, nbs)
            clusters_selected[nb_list] .= true

            if length(nb_list) < model.min_pts
                empty!(nb_list)
                continue
            end

            for i in nb_list
                visited[i] && continue
                push!(FIFO, i)
            end

            empty!(nb_list)
        end
        
        ind_cluster = findall(clusters_selected)
        model.min_pts ≤ sum(clusters_selected) && push!(model.clusters, ind_cluster)
    end

    nc = 1
    for cl in model.clusters
        for i in cl
            model.labels[i] = nc
        end
        nc += 1
    end
end