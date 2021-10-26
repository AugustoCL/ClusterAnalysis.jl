# push!(LOAD_PATH,"../src/")
using Documenter, ClusterAnalysis

DocMeta.setdocmeta!(ClusterAnalysis, :DocTestSetup, :(using ClusterAnalysis); recursive=true)

makedocs(
    modules=[ClusterAnalysis],
    authors="AugustoCL <augustoleal72@gmail.com>, Elias Carvalho <eliascarvdev@gmail.com>",
    repo="https://github.com/AugustoCL/ClusterAnalysis.jl",
    sitename="ClusterAnalysis.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://AugustoCL.github.io/ClusterAnalysis.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Algorithms" => [
            "K-Means" => "algorithms/kmeans.md",
            "DBSCAN" => "algorithms/dbscan.md"
        ],
        "API" => [
            "Functions" => "API/functions.md", 
            "Types" => "API/types.md"
        ]
    ]
)

deploydocs(
    repo="github.com/AugustoCL/ClusterAnalysis.jl.git",
    devbranch = "main"
)