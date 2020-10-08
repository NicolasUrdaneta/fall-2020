##                                                                            ##
##              Problem set #2 for Structural modelling course                ##
##                               Nicolás Urdaneta                             ##
##                                                                            ##

clearconsole()
#=
Pkg.add("Optim")
Pkg.add("HTTP")
Pkg.add("GLM")
=#

using LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, Distributions, Optim, HTTP, GLM

## 1. Basic optimization in Julia

f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
negf(x) =-copy(f(x))
startval = rand(1)
result = optimize(negf, startval, LBFGS())
println(result)

## 2. Optimization with data

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
	ssr = (y.-X*beta)'*(y.-X*beta)
	return ssr
end

beta_hat_ols = optimize(b-> ols(b,X,y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations = 100_000, show_trace = true))

println(beta_hat_ols.minimizer)

	## Just checking:

	bols = inv(X'X)*X'y
	df.white = df.race.==1
	bols_lm = lm(@formula(married ~ age+white+collgrad), df)

## 3. Optimization logit lokelihood

# Log likelihood
# y(lnP) + (1-y)ln(1-P)

function logit_ll(alpha, X, Y)

	ll_i = (y.==1).*log.(exp.(X*alpha)./(1 .+exp.(X*alpha))).+(y.==0).*log.(1 ./(1 .+exp.(X*alpha)))

	ll = -sum(ll_i)

	return ll

end

alpha_hat_optim = optimize(a -> logit_ll(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(alpha_hat_optim.minimizer)


## 4. Check logit using glm

alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println(alpha_hat_glm)


## 5. Multinomial logit

	 # Join categories
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

	# Redefine X and Y as we changed their rows
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, Y)

	K = size(X,2)
	J = length(unique(y))
	N = size(X,1)

	bigY = zeros(N,J)
	for j=1:J
		bigY[:,j] = y.==j
	end

	bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

	num = zeros(N,J)
	dem = zeros(N)
	for j=1:J
		num[:,j] = exp.(X*bigAlpha[:,j])
		dem .+= num[:,j]
	end

	P = num ./ repeat(dem,1,J)

	ll = -sum( bigY.*log.(P))

end

alpha_zero = zeros(6*size(X,2))
alpha_rand = rand(6*size(X,2))
alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]

alpha_start = alpha_true.*rand(size(alpha_true))
println(size(alpha_true))

alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)

## Stanrd erorrs

	## Adjust mlogit function
function mlogit_hessian(alpha, X, Y)

	K = size(X,2)
	J = length(unique(y))
	N = size(X,1)

	bigY = zeros(N,J)
	for j=1:J
		bigY[:,j] = y.==j
	end

	bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

	T = promote_type(eltype(X),eltype(alpha)) # this line is new
	num = zeros(T,N,J) ## Added T first
	dem = zeros(T,N) ## Added T first
	for j=1:J
		num[:,j] = exp.(X*bigAlpha[:,j])
		dem .+= num[:,j]
	end

	P = num ./ repeat(dem,1,J)

	ll = -sum( bigY.*log.(P))

end

# declare that the objective function is twice differentiable
td = TwiceDifferentiable(b -> mlogit_for_h(b, X, y), alpha_start; autodiff = :forward)
# run the optimizer
alpha_hat_optim_ad = optimize(td, alpha_zero, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_mle_ad = alpha_hat_optim_ad.minimizer
# evaluate the Hessian at the estimates
H  = Optim.hessian!(td, alpha_hat_mle_ad)
# standard errors = sqrt(diag(inv(H))) [usually it's -H but we've already multiplied the obj fun by -1]
alpha_hat_mle_ad_se = sqrt.(diag(inv(H)))
println([alpha_hat_mle_ad alpha_hat_mle_ad_se]) # these standard errors match Stata
