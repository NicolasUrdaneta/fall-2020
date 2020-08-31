##                                                                            ##
##              Problem set #1 for Structural modelling course                ##
##                               Nicolás Urdaneta                             ##
##                                                                            ##


## Install & load packages

Pkg.add("JLD2")
Pkg.add("Random")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("CSV")
Pkg.add("FreqTables")
Pkg.add("DataFrames")
Pkg.add("Distributions")


using JLD2
using Random
using LinearAlgebra
using Statistics
using CSV
using FreqTables
using DataFrames
using Distributions

## 0. Github account
    ## Nurdaneta
    ## Forked

## 1. Initializing variables and basic matrix operations

    # a) Matrices of random numbers

Random.seed!(1234)

A = rand(Uniform(-5,10), 10,7)
B = rand(Normal(-2,15), 10,7)
C = [A[1:5,1:5] B[1:5,6:7]]
D = copy(A)
D[D.>0] .= 0

    # b) List number elements of A
println(A)

    # c) List unique elements of D
println(unique(D))

    # d) reshape function, create E: vec operator to B
E = reshape(B, (70,1))
E = vec(B)

    # e) 3 dimensional matrix
F = cat(A,B;dims=3)

    # f)
#F = permutedims(F,2)

    # g) kronecker product
G = kron(B,C)
# Will give error: G = kron(C,F)

    # h)



    ## 2. Loops and comprehensions

    # a)
AB = Array{Float64}(undef, 10, 7)
for i in 1:10
    for j in 1:7
        AB[i,j] = A[i,j]*B[i,j]
    end
end

AB2 = A.*B

    # b)
Cprime = [C[i] for i in eachindex(C) if (C[i]<=5 && C[i]>=-5)]

    # c)
X = Array{Float64}(undef, 15169, 6, 5)
X[1:15169,1,1:5] .= 1

for i in 1:5
    X[1:15169,2,i] = rand(Uniform(0,1), 15169,1)
    X[1:15169,2,i]  = (X[1:15169,2,i] .< 0.75*(6-i)/5)
end

#Y = copy(X)
#Y[1:15169,2,1]  = (Y[1:15169,2,1] .< 0.75)

for i in 1:5
    X[1:15169,3,i] = rand(Normal(15+i-1,5(i-1)), 15169,1)
end

for i in 1:5
    X[1:15169,4,i] = rand(Normal(pi*(6-i)/3,1/exp(1)), 15169,1)
end

for i in 1:5
    X[1:15169,5,i] = rand(Binomial(20,0.6),15169)
end

for i in 1:5
    X[1:15169,6,i] = rand(Binomial(20,0.5),15169)
end

    # d)
β =  Array{Float64}(undef, 6, 5)
β[1,1:5] = [(1+0.25*(i-1)) for i in 1:5 ]
β[2,1:5] = [log(i) for i in 1:5]
β[3,1:5] = [-sqrt(i) for i in 1:5]
β[4,1:5] = [exp(i)-exp(i+1) for i in 1:5]
β[5,1:5] = [i for i in 1:5]
β[6,1:5] = [i/3 for i in 1:5]

    # e)
Y =  Array{Float64}(undef, 15169, 6)

Y = dot(X,β)


## 3. Reading in Data and calculating summary statistics

    # a)
using CSV; data = CSV.read("nlsw88.csv")
@save "nlsw88.jld2" data
#using JLD; save("nlsw88.jld", "data", data)

    # b)
describe(data[!,"married"])

    # c)
freqtable(data[!,"race"])

    # d)
summary = describe(data, :mean, :std, :median, :min, :max, :nunique, :q75, :q25)

    # e)
freqtable(data[!,"industry"],data[!,"occupation"])

## 4. Practice with functions
