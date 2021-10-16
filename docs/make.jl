using Documenter, ClusterAnalysis

makedocs(
    modules=[ClusterAnalysis],
    format=Documenter.HTML(prettyurls=false),
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
    ],
    repo="https://github.com/AugustoCL/ClusterAnalysis.jl",
    sitename="ClusterAnalysis.jl",
    authors="AugustoCL <augustoleal72@gmail.com>, Elias Carvalho <eliascarvdev@gmail.com>"
)

deploydocs(
    repo="github.com/AugustoCL/ClusterAnalysis.jl.git",
    devbranch = "main"
)