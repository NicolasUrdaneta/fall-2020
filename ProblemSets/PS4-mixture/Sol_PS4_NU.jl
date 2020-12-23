##                                                                            ##
##              Problem set #4 for Structural modelling course                ##
##                               Nicolás Urdaneta                             ##
##                                                                            ##

using LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, Optim, HTTP, GLM, ForwardDiff


url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

## 1. multinomial logit as PS3

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

## Optimize. Not recommended to use random start values (it could take voer 30 minutes. These are the results from PS3)

## startvals = [2*rand(7*size(X,2)).-1; .1]
startvals = [ .0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; .1168824; -.2870554; -5.322248; 1.307477]


## define as twice differentiable
td = TwiceDifferentiable(theta-> mlogit_Z(theta,X,Z,y), startvals; autodiff = :forward)

## Run

@time theta_hat_optim = optimize(td, startvals, LBFGS(), Optim.Options(g_tol=1e-5, iterations = 100_000, show_trace = true, show_every=50))

theta_hat_mle = theta_hat_optim.minimizer
H  = Optim.hessian!(td, theta_hat_mle)
theta_hat_mle_se = sqrt.(diag(inv(H)))
println([theta_hat_mle theta_hat_mle_se])


## 3. Mixed logit

# a) Quadrature axample normal
d = Normal(0,1)

nodes, weights = lgwt(7,-4,4)

sum(weights.*pdf.(d,nodes))
sum(weights.*nodes.*pdf.(d,nodes))

## b) More examples

# i)
d = Normal(0,2)
nodes, weights = lgwt(7,-10,10)
sum(weights.*nodes.*nodes.*pdf.(d,nodes))

# ii)
d = Normal(0,2)
nodes, weights = lgwt(10,-10,10)
sum(weights.*nodes.*nodes.*pdf.(d,nodes))

#These are aproximations to the variance, should get close to 4

## c) monte Carlo integration

# i)
b = 10
a = -10
c = Uniform(a,b)
draws = rand(c,1_000_000)
println((b-a)*mean(draws.^2 .*pdf.(d,draws)))
println((b-a)*mean(draws .*pdf.(d,draws)))
println((b-a)*mean(pdf.(d,draws)))

## 4. Question 1 code to optimize the expression using quadrature
	## Caution: do not try to run

function mlogit_Z_Quad(theta, X, Z, y, R)

		alpha = theta[1:end-2]
		gamma = theta[end-1]
		sigma = exp(theta[end]) # makes sure sigma is never 0 (which would make the Normal density undefined)

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

		# Write probability as a function of variable of integration
		function ll_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R)
			c = Uniform(-4*sigma,4*sigma)
			draws = rand(c,R)
			#nodes, weights = lgwt(R,-4*sigma,4*sigma)
			out = zeros(T,N)

			for r=1:R
				num = zeros(T,N,J)
				den = zeros(T,N)
				Prob = zeros(T,N,J)
				for j=1:J
					num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*nodes[r])
					den .+= num[:,j]
				end
				Prob = num./repeat(den,1,J)
				out .+= vec( weights[r]*(prod(Prob.^bigY; dims=2))*pdf(Normal(gamma,sigma), nodes[r]))
				# We use vec because vectors in Julia are "flat" and this operator flattens a one dimensional array.
			end
			return out
		end

		intll = ll_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R)
		ll = -sum(log.(intll))

		return ll

end

## 5. Question 1 code to optimize the expression using Monte Carlo simulation
	## Caution: do not try to run

	function mlogit_Z_MonteC(theta, X, Z, y, R)

			alpha = theta[1:end-2]
			gamma = theta[end-1]
			sigma = exp(theta[end]) # makes sure sigma is never 0 (which would make the Normal density undefined)

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

			# Write probability as a function of variable of integration
			function ll_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R)
				nodes, weights = lgwt(R,-4*sigma,4*sigma)
				out = zeros(T,N)

				for r=1:R
					num = zeros(T,N,J)
					den = zeros(T,N)
					Prob = zeros(T,N,J)
					for j=1:J
						num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*nodes[r])
						den .+= num[:,j]
					end
					Prob = num./repeat(den,1,J)
					out .+= vec( (8*sigma/R)*(prod(Prob.^bigY; dims=2))*pdf(Normal(gamma,sigma), draws[r]))
					# We use vec because vectors in Julia are "flat" and this operator flattens a one dimensional array.
				end
				return out
			end

			intll = ll_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R)
			ll = -sum(log.(intll))

			return ll

	end
