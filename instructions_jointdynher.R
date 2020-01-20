library(Rcpp)
library(RcppEigen)


#To compile C++ code on R, a compiler is needed. For Windows, the compiler comes with Rtools
#For further information, see https://support.rstudio.com/hc/en-us/articles/200486498

Sys.setenv("PKG_CXXFLAGS"="-fopenmp -std=c++11 -O3 -DEIGEN_NO_DEBUG")
Sys.setenv("PKG_LIBS"="-fopenmp") 					#Compiler flags
setwd("C:/Users/Arttu/Documents/GitHub/aarjas.github.io")	#Sets working directory to where the code is
										#Change this to the right one

sourceCpp("jointdynher.cpp") 						#Compiles the code

#Create some data
n=500;t=50
G=matrix(rnorm(n^2),n,n)
G=G%*%t(G)/n+0.1*diag(n)						#Relationship matrix
eG=eigen(G)
ksi=eG$values 								#Eigenvalues of G
U=eG$vectors 								#Eigenvectors of G
x=seq(0,2*pi,len=t) 							#Measurement grid
se=sin(x)+2 								#Environmental variance
sg=cos(x)+2 								#Genetic variance
y=matrix(0,n,t)
for(i in 1:t)
{
	lambda=sg[i]*ksi+se[i]
	y[,i]=(U%*%diag(sqrt(lambda))%*%t(U))%*%rnorm(n)
}

Nsim=10000									#Sets the number of MCMC-iterations
linit=t/3 									#Initial values for the length scales. t/3 is usually fine
cores=parallel::detectCores() 					#Number of processor cores

#The simulation function takes 5 parameters: the data, number of iterations, the eigenvectors and eigenvalues of the
#relationship matrix and the number of processor cores. The algorithm usually runs the fastest with half of the total cores.
#The data must be a matrix with rows corresponding to the individuals and columns corresponding to the time points

C=sim(y,Nsim,U,ksi,cores/2)

#Examine the results
meanse=colMeans(C$ser)							#Environmental variance posterior mean
meansg=colMeans(C$sgr)							#Genetic variance posterior mean
boundsse=apply(C$ser,2,quantile,c(0.025,0.975))			#95% credible interval of env. variance
boundssg=apply(C$sgr,2,quantile,c(0.025,0.975)) 		#95% credible interval of gen. variance
her=C$sgr/(C$sgr+C$ser) 						#Compute heritability
meanher=colMeans(her)							#Posterior mean of heritability
boundsher=apply(her,2,quantile,c(0.025,0.975)) 			#95% credible interval of heritability
paste(round(C$Time_loop/3600*60,1), "minutes")			#Computation time

#Plotting
par(mfrow=c(3,1))
plot(x,meanse,type="l",ylim=c(min(boundsse),max(boundsse)))
apply(boundsse,1,lines,lty=2,x=x)
plot(x,meansg,type="l",ylim=c(min(boundssg),max(boundssg)))
apply(boundssg,1,lines,lty=2,x=x)
plot(x,meanher,type="l",ylim=c(min(boundsher),max(boundsher)))
apply(boundsher,1,lines,lty=2,x=x)

#Examine convergence from traceplots
par(mfrow=c(2,2))
plot(C$ser[,sample(t,1)],type="l")					#Random env. variance element
plot(C$sgr[,sample(t,1)],type="l")					#Random gen. variance element
plot(C$le,type="l")							#Env. length scale
plot(C$lg,type="l")							#Gen. length scale
