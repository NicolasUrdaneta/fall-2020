##                                                                            ##
##              Problem set #1 for Structural modelling course                ##
##                               Nicolás Urdaneta                             ##
##                                                                            ##


## Install & load packages

#=
Pkg.add("JLD")
Pkg.add("JLD2")
Pkg.add("Random")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("CSV")
Pkg.add("FreqTables")
Pkg.add("DataFrames")
Pkg.add("Distributions")
=#
#Pkg.add("HDF5")
using JLD
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

function q1()
    # i.-iv. create matrices
    A = rand(Uniform(-5,10), 10,7)
    # Or A = 15*rand(10,7).-5
    B = rand(Normal(-2,15), 10,7)
    # Or B = 15*randn(10,7).-2
    C = [A[1:5,1:5] B[1:5,6:7]]
    # Or C = cat(A[1:5,1:5],B[1:5,end-1:end]; dims=2)
    D = copy(A)
    D[D.>0] .= 0
    # Or D = A.*(A.<=0)

    # b) List number elements of A
    println(length(A))

    # c) List unique elements of D
    println(length(unique(D)))

    # d) reshape function, create E: vec operator to B
    E = reshape(B, (70,1))
    E = vec(B)
        # Or E = reshape(B,length(B))
        # Or E = B[:]

    # e) 3 dimensional matrix
    F = cat(A,B;dims=3)

    # f) permute
    F = permutedims(F,[3 1 2])

    # g) kronecker product
    G = kron(B,C)
    # Will give error: G = kron(C,F)

    # h) saving
    save("matrixpractice.jld","A",A,"B",B,"C",C,"D",D,"E",E,"F",F,"G",G)

    # i) saving
    save("firstmatrix.jld","A",A,"B",B,"C",C,"D",D)

    # j) export to csv
    CSV.write("Cmatrix.csv",DataFrame(C))

    # export to DAT
    CSV.write("Dmatrix.dat",DataFrame(D); delim="\t")

    # Function around code
    return A,B,C,D
end

    ## 2. Loops and comprehensions

    function q2(A,B,C)
        # a)
    AB = Array{Float64}(undef, 10, 7)
    for i in 1:10
        for j in 1:7
            AB[i,j] = A[i,j]*B[i,j]
        end
    end
    # OR:     AB = [A[i,j]*B[i,j] for i=1:size(A,1),j=1:size(A,2)]

    AB2 = A.*B

        # b)
    Cprime = [C[i] for i in eachindex(C) if (C[i]<=5 && C[i]>=-5)]
    Cprime2 = C[(C.>=-5) .& (C.<=5)]
        # c)
    X = Array{Float64}(undef, 15169, 6, 5)
    X[1:15169,1,1:5] .= 1

    for i in 1:5
        X[1:15169,2,i] = rand(Uniform(0,1), 15169,1)
        X[1:15169,2,i]  = (X[1:15169,2,i] .< 0.75*(6-i)/5)

        X[1:15169,3,i] = rand(Normal(15+i-1,5(i-1)), 15169,1)

        X[1:15169,4,i] = rand(Normal(pi*(6-i)/3,1/exp(1)), 15169,1)

        X[1:15169,5,i] = rand(Binomial(20,0.6),15169)

        X[1:15169,6,i] = rand(Binomial(20,0.5),15169)
    end

    #= OR:
    N = 15_169
    K = 6
    T = 5
    X = cat([cat([ones(N,1) rand(N,1).<=(0.75*(6-t)/5) (15+t-1).+(5*(t-1)).*randn(N,1) (π*(6-t)/3).+(1/exp(1)).*randn(N,1) rand(Binomial(20,0.6),N) rand(Binomial(20,0.5),N)];dims=3) for t=1:T]...;dims=3) # discrete_normal binomial
    =#

        # d)
    β =  Array{Float64}(undef, 6, 5)
    β[1,1:5] = [(1+0.25*(i-1)) for i in 1:5 ]
    β[2,1:5] = [log(i) for i in 1:5]
    β[3,1:5] = [-sqrt(i) for i in 1:5]
    β[4,1:5] = [exp(i)-exp(i+1) for i in 1:5]
    β[5,1:5] = [i for i in 1:5]
    β[6,1:5] = [i/3 for i in 1:5]

    #= OR
    β = vcat([cat([1+0.25*(t-1) log(t) -sqrt(t) exp(t)-exp(t+1) t t/3];dims=1) for t=1:T]...)'
    =#

        # e)
    Y = hcat([cat(X[:,:,t]*β[:,t] + .36*randn(N,1);dims=2) for t=2:T]...)

    return nothing
end


## 3. Reading in Data and calculating summary statistics

function q3()

        # a)
    using CSV; data = CSV.read("nlsw88.csv")
    #save("nlsw88.jld","nlsw88",nlsw88)
    @save "nlsw88.jld2" data
    #using JLD; save("nlsw88.jld", "data", data)

        # b)
    mean(data.never_married)
    mean(data.collgrad)

        # c)
    freqtable(data[!,"race"])
    freqtable(data, :race)

        # d)
        summarystats = describe(data)

        # e)
    freqtable(data[!,"industry"],data[!,"occupation"])
    freqtable(data, :industry, :occupation)

        # f)
    wageonly = data[:,[:industry, :occupation,:wage]]
    grouper = groupby(wageonly, [:industry,:occupation])
    combine(grouper, valuecols(grouper) .=> mean)

    return nothing
end

## 4. Practice with functions

funciton q4()

    # a)
    mats = load("firstmatrix.jld")
    A = mats["A"]
    B = mats["B"]
    C = mats["C"]
    D = mats["D"]

    # b) 
    function matrixops(m1,m2)

        if size(m1)!=size(m2)
            error("inputs must have the same size.")
        end

        r1 = m1.*m2
        r2 = m1'*m2
        r3 = sum(m1)+sum(m2)

        return r1, r2, r3

    end

    # d)
    matrixops(A,B)

    # e)
    matrixops(C,D)

    # f)
    matrixops(convert(Array, data.ttl_exp), convert(Array, data.wage))

    return nothing
end
