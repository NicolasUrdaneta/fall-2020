##                                                                            ##
##              Problem set #3 for Structural modelling course                ##
##                               Nicolás Urdaneta                             ##
##                                                                            ##
#Pkg.add("ForwardDiff")

using LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, Optim, HTTP, GLM, ForwardDiff

## 1. multinomial logit with alternative-specific covariates Z

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
		 df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

function mlogit_Z(theta, X, Z, y)

	alpha = theta[1:end-1]
	gamma = theta[end]

	K = size(X,2)
	J = length(unique(y))
	N = length(y)
	bigY = zeros(N,J)

	# Matrix with a column equal 1 if y equals that alternative
	for j=1:J
		bigY[:,j] = y.==j
	end

	## Matrix of parameters, normalice last alternative to zero
	bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

	T = promote_type(eltype(X),eltype(theta))
	num   = zeros(T,N,J)
	den   = zeros(T,N)

	for j=1:J
		num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
		den = den .+ num[:,j]
	end

	Prob = num./repeat(den,1,J)

	ll = -sum(bigY.*log.(Prob))

	return ll

end

## Optimize

startvals = [2*rand(7*size(X,2)).-1; .1]

## define as twice differentiable
td = TwiceDifferentiable(theta-> mlogit_Z(theta,X,Z,y), startvals; autodiff = :forward)

## Run

theta_hat_optim = optimize(td, startvals, LBFGS(), Optim.Options(g_tol=1e-5, iterations = 100_000, show_trace = true, show_every=50))

theta_hat_mle = theta_hat_optim.minimizer



# The coefficient gamma represents the change in utility with a 1-	unit change in log wages
# More properly, gamma/100 is the change in utility with a 1% increase in expected wage



## 3. Nested logit with the following structure

	# White collar ocupations: Professional, Managers, Sales
	# Blue collar occupations: Unskilled, Craftsmen, Operatives, Transport
	# Other occupations

	# Parameters such that they are only nest-level (not alternative level):
	  # β_WC; β_BC; λ_WC; λ_BC; γ; β_other = normalized  to 0


function nested_mlogit(theta, X, Z, y, nests)

	beta = theta[1:end-3] ## One coefficient for each variable in each nest
	lambda = theta[end-2:end-1] ## One for each nest
	gamma = theta[end] ## Only one

	K = size(X,2)
	J = length(unique(y))
	N = length(y)
	bigY = zeros(N,J)

	# Matrix with a column equal 1 if y equals that alternative
	for j=1:J
		bigY[:,j] = y.==j
	end

	## Matrix of parameters, normalice last alternative to zero
	bigbeta = [repeat(beta[1:K],1,length(nests[1])) repeat(beta[K+1:2K],1,length(nests[2])) zeros(K)]

	T = promote_type(eltype(X),eltype(theta))
	num   = zeros(T,N,J)
	den   = zeros(T,N)
	first_num = zeros(T,N,J)

	## Create first part of numerator
	for j=1:J
		if j in nests[1]
			first_num[:,j] = exp.( (X*bigbeta[:,j] .+ (Z[:,j].-Z[:,J])*gamma)./lambda[1])
		elseif j in nests[2]
			first_num[:,j] = exp.( (X*bigbeta[:,j] .+ (Z[:,j].-Z[:,J])*gamma)./lambda[2])
		else
			first_num[:,j] = exp.(zeros(N))
		end
	end

	## Create second part of numerator, numerator, and denominator
	for j = 1:J
		if j in nests[1]
			num[:,j] = first_num[:,j] .* (sum(first_num[:,nests[1][:]], dims=2)).^(lambda[1]-1)
		elseif j in nests[2]
			num[:,j] = first_num[:,j] .* (sum(first_num[:,nests[2][:]], dims=2)).^(lambda[2]-1)
		else
			num[:,j] = first_num[:,j]
		end
		den .+= num[:,j]
	end


	Prob = num./repeat(den,1,J)
	ll = -sum(bigY.*log.(Prob))

	return ll

end

## Optimize

startvals = [2*rand(2*size(X,2)).-1; 1; 1; .1]
nests = [[1 2 3], [4 5 6 7]]

## define as twice differentiable
td = TwiceDifferentiable(theta-> nested_mlogit(theta,X,Z,y, nests), startvals; autodiff = :forward)

## Run

@time nested_logit_optim = optimize(td, startvals, LBFGS(), Optim.Options(g_tol=1e-5, iterations = 100_000, show_trace = true, show_every=50))

nested_logit_mle = nested_logit_optim.minimizer
H = Optim.hessian!(td, nested_logit_mle)
nested_logit_mle_se = sqrt.(diag(inv(H)))

println(nested_logit_mle, nested_logit_mle_se)
