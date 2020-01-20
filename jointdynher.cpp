#define _USE_MATH_DEFINES

#include <cmath>
#include <RcppEigen.h>
#include <ctime>
#include <omp.h>
#include <random>
#include <string>

// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;

const double pi = M_PI;
std::default_random_engine generator;

// [[Rcpp::export]]
MatrixXd matern(const double& l, const int& n)
{
    MatrixXd C(n, n);
    #pragma omp parallel for
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i <= j; i++)
        {
            const int d = abs(i - j);
            C(i, j) = C(j, i) = (1 + sqrt(5)*d/l + 5*d*d/(3*l*l))*exp(-sqrt(5)*d/l);
        }
    }
    return C;
}

// [[Rcpp::export]]
VectorXd randomnormal(const int& n)
{
    std::normal_distribution<double> distribution(0, 1);
    VectorXd r(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
    {
        r(i) = distribution(generator);
    }
    return r;
}

// [[Rcpp::export]]
double randomnormal1()
{
    std::normal_distribution<double> distribution(0, 1);
    return distribution(generator);
}

double randomuniform(const double& lower, const double& upper)
{
    std::uniform_real_distribution<> distribution(lower, upper);
    return distribution(generator);
}

// [[Rcpp::export]]
double normaldensity(const double& x, const double& mean, const double& sd)
{
    double xmean = x - mean;
    double xmean2 = xmean*xmean;
    return -log(sd) - 0.5*xmean2/(sd*sd);
}

double var(const ArrayXd& x, const int& n)
{
    const double mean = x.sum()/n;
    return ((x - mean).pow(2)).sum()/(n - 1);
}


// [[Rcpp::export]]
SparseMatrix<double> H(const double& l, const double& n)
{
    SparseMatrix<double> C(n, n);
    double l2 = l*l;
    C.insert(0, 0) = 1 + 2*l2;
	C.insert(0, 1) = -l2;
	//C.insert(0, n - 1) = -l2;
    for(int i = 1; i < n - 1; i++)
    {
        C.insert(i, i) = 1 + 2*l2;
		C.insert(i, i - 1) = -l2;
		C.insert(i, i + 1) = -l2;
    }
    //C.insert(n - 1, 0) =  -l2;
    C.insert(n - 1, n - 2) = -l2;
    C.insert(n - 1, n - 1) = 1 + 2*l2;
    C.makeCompressed();
    return C/sqrt(4*l);
}


//Plotting
// [[Rcpp::export]]
void plot_r_cpp_call(const VectorXd& x, const std::string& xlab, const std::string& main){

  // Obtain environment containing function
  Rcpp::Environment graph("package:graphics");

  // Make function callable from C++
  Rcpp::Function plot_r = graph["plot"];

  // Call the function and receive its list output
  plot_r(Rcpp::_["x"] = x, Rcpp::_["type"]  = "l", Rcpp::_["ylab"]  = "", Rcpp::_["xlab"]  = xlab, Rcpp::_["main"]  = main); // example of additional param

}

// [[Rcpp::export]]
void par_r_cpp_call(const VectorXd& n){

  // Obtain environment containing function
  Rcpp::Environment gr("package:graphics");

  // Make function callable from C++
  Rcpp::Function par_r = gr["par"];

  // Call the function and receive its list output
  par_r(Rcpp::_["mfrow"] = n); // example of additional param

}

// [[Rcpp::export]]
Rcpp::List sim(MatrixXd Y, const int& Nsim, const MatrixXd& U, const ArrayXd& ksi, const int& cores)
{
	struct timespec start, init, finish;
	double elapsed_init, elapsed_loop;
	clock_gettime(CLOCK_MONOTONIC, &start);
    omp_set_num_threads(cores);
	int mod = (Nsim + 0.0)/100;

	//Number of individuals and time points
    const int n = Y.col(0).size();
    const int t = Y.row(0).size();

	//Plotting
	VectorXd plot_row_col(2);
	plot_row_col << 2, 1;
	par_r_cpp_call(plot_row_col);

	//Scale data
	VectorXd colsds(t);
	ArrayXXd Ya = Y.array();
	for(int i = 0; i < t; i++)
    {
        Ya.col(i) -= Ya.col(i).mean();
        colsds(i) = sqrt(var(Ya.col(i), n));
    }
    double scale = sqrt(2)/colsds.mean();
    Ya = scale*Ya;
    Y = Ya.matrix();

	Rcpp::Rcout << "Initializing chains, decomposing G...\n\n";


    const MatrixXd Yt = Y.transpose();

	const ArrayXXd z2 = (U.transpose()*Y).array().square();

    //Parameter chain initialization

    //Whitened variance chains
    const int extn = round((t + 0.0)/4);
    MatrixXd se = MatrixXd::Zero(Nsim, t + 2*extn);
    MatrixXd sg = MatrixXd::Zero(Nsim, t + 2*extn);

    //Real variance chains
    MatrixXd ser = MatrixXd::Zero(Nsim, t + 2*extn);
    MatrixXd sgr = MatrixXd::Zero(Nsim, t + 2*extn);

    //Length scale chains
    VectorXd le(Nsim); le(0) = log((t + 0.0)/3);
    VectorXd lg(Nsim); lg(0) = log((t + 0.0)/3);

    //GP matrices
    SimplicialLDLT<SparseMatrix<double> > solvere;
    SimplicialLDLT<SparseMatrix<double> > solverg;
    SparseMatrix<double> He = H(exp(le(0)), t + 2*extn);
    SparseMatrix<double> Hg = H(exp(lg(0)), t + 2*extn);
    solvere.analyzePattern(He);
    solvere.factorize(He);
    solverg.analyzePattern(Hg);
    solverg.factorize(Hg);
    VectorXd cse = solvere.solve(se.row(0).transpose());
    VectorXd csg = solverg.solve(sg.row(0).transpose());

    //Length scale prior parameters
    MatrixXd X(2, 2);
    X << 1, -1.96,  1, 1.96;
    VectorXd logab(2);
    logab << 0, log(t - 1);
    VectorXd hyperparams = X.colPivHouseholderQr().solve(logab);
    double mul = hyperparams(0);
    double taul = hyperparams(1);

    //Likelihood evaluation
    double like = 0;
    #pragma omp parallel for reduction(+:like)
    for(int i = 0; i < t; i++)
    {
        ArrayXd lambda = ksi*exp(csg(i + extn)) + exp(cse(i + extn));
        double ldK = log(lambda).sum();
        like += -0.5*ldK - 0.5*(z2.col(i)*inverse(lambda)).sum();
    }

    //Recursive means
    ArrayXd itermeane = cse;
    ArrayXd itermeang = csg;

    //Adaptive
    int batch = 0;
    int lase = 0;
    int lasg = 0;
    int lasee = 0;
    int lasgg = 0;
    double sle = 0;
    double slg = 0;

	clock_gettime(CLOCK_MONOTONIC, &init);

	Rcpp::Rcout << "Initialization done, starting MCMC loop...\n\n";

    //Simulation loop
    for(int i = 1; i < Nsim; i++)
    {
        //Whitened environmental variance update
        VectorXd w = randomnormal(t + 2*extn);
        like += log(randomuniform(0, 1));
        double theta = randomuniform(0, 2*pi);
        double thetamin = theta - 2*pi;
        double thetamax = theta;
        double likep; VectorXd csep;
        do
        {
            VectorXd sep = se.row(i - 1).transpose()*cos(theta) + w*sin(theta);
            csep = solvere.solve(sep);
            likep = 0;
            #pragma omp parallel for reduction(+:likep)
			for(int j = 0; j < t; j++)
			{
				ArrayXd lambda = ksi*exp(csg(j + extn)) + exp(csep(j + extn));
				double ldK = log(lambda).sum();
				likep += -0.5*ldK - 0.5*(z2.col(j)*inverse(lambda)).sum();
			}
            if(likep > like)
                se.row(i) = sep.transpose();
            else
            {
                if(theta < 0)
                    thetamin = theta;
                else
                    thetamax = theta;
                theta = randomuniform(thetamin, thetamax);
            }
        }
        while(likep <= like);
        like = likep;
        cse = csep;

        //Whitened genetic variance update
        w = randomnormal(t + 2*extn);
        like += log(randomuniform(0, 1));
        theta = randomuniform(0, 2*pi);
        thetamin = theta - 2*pi;
        thetamax = theta;
        VectorXd csgp;
        do
        {
            VectorXd sgp = sg.row(i - 1).transpose()*cos(theta) + w*sin(theta);
            csgp = solverg.solve(sgp);
            likep = 0;
            #pragma omp parallel for reduction(+:likep)
            for(int j = 0; j < t; j++)
			{
				ArrayXd lambda = ksi*exp(csgp(j + extn)) + exp(cse(j + extn));
				double ldK = log(lambda).sum();
				likep += -0.5*ldK - 0.5*(z2.col(j)*inverse(lambda)).sum();
			}
            if(likep > like)
                sg.row(i) = sgp.transpose();
            else
            {
                if(theta < 0)
                    thetamin = theta;
                else
                    thetamax = theta;
                theta = randomuniform(thetamin, thetamax);
            }
        }
        while(likep <= like);
        like = likep;
        csg = csgp;

        //Environmental variance length scale update
        double lep = le(i - 1) + exp(sle)*randomnormal1();
        SparseMatrix<double> Hep = H(exp(lep), t + 2*extn);
        solvere.factorize(Hep);
		csep = solvere.solve(se.row(i).transpose());
        likep = 0;
        #pragma omp parallel for reduction(+:likep)
        for(int j = 0; j < t; j++)
		{
			ArrayXd lambda = ksi*exp(csg(j + extn)) + exp(csep(j + extn));
            double ldK = log(lambda).sum();
            likep += -0.5*ldK - 0.5*(z2.col(j)*inverse(lambda)).sum();
		}
		double ale = likep - like + normaldensity(lep, mul, taul) - normaldensity(le(i - 1), mul, taul) ;
        if(log(randomuniform(0, 1)) < ale)
        {
            le(i) = lep;
            like = likep;
            cse = csep;
            He = Hep;
            lase++;
            lasee++;
        }
        else
        {
            le(i) = le(i - 1);
            solvere.factorize(He);
        }
        if(i % 50 == 0)
        {
             batch++;
             double delta = std::min(0.01, 1.0/sqrt(i));
             if((lase + 0.0)/50 < 0.44)
                sle -= delta;
             else
                sle += delta;
            lase = 0;
        }

        //Genetic variance length scale update
        double lgp = lg(i - 1) + exp(slg)*randomnormal1();
        SparseMatrix<double> Hgp = H(exp(lgp), t + 2*extn);
        solverg.factorize(Hgp);
		csgp = solverg.solve(sg.row(i).transpose());
        likep = 0;
        #pragma omp parallel for reduction(+:likep)
        for(int j = 0; j < t; j++)
		{
			ArrayXd lambda = ksi*exp(csgp(j + extn)) + exp(cse(j + extn));
            double ldK = log(lambda).sum();
            likep += -0.5*ldK - 0.5*(z2.col(j)*inverse(lambda)).sum();
		}
		double alg = likep - like + normaldensity(lgp, mul, taul) - normaldensity(lg(i - 1), mul, taul) ;
        if(log(randomuniform(0, 1)) < alg)
        {
            lg(i) = lgp;
            like = likep;
            csg = csgp;
            Hg = Hgp;
            lasg++;
            lasgg++;
        }
        else
        {
            lg(i) = lg(i - 1);
            solverg.factorize(Hg);
        }
        if(i % 50 == 0)
        {
            double delta = std::min(0.01, 1.0/sqrt(i));
             if((lasg + 0.0)/50 < 0.44)
                slg -= delta;
             else
                slg += delta;
            lasg = 0;
        }





        //Real variances
        ser.row(i) = cse.transpose();
		sgr.row(i) = csg.transpose();

		itermeane = (i + 0.0)/(i + 1.0)*itermeane + 1.0/(i + 1.0)*cse.array();
		itermeang = (i + 0.0)/(i + 1.0)*itermeang + 1.0/(i + 1.0)*csg.array();

		if(i % 500 == 0)
        {
            plot_r_cpp_call(exp(itermeane.segment(extn, t) - 2*log(scale)), "Environmental variance", "Cumulative means of parameters");
			plot_r_cpp_call(exp(itermeang.segment(extn, t) - 2*log(scale)), "Genetic variance", "");
        }



		if(i % mod == 0)
        {
            Rcpp::Rcout << (i + 0.0)/Nsim*100 << "% completed\n\n";
        }
    }


	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_init = (init.tv_sec - start.tv_sec);
	elapsed_loop = (finish.tv_sec - init.tv_sec);
	elapsed_init += (init.tv_nsec - start.tv_nsec) / 1000000000.0;
	elapsed_loop += (finish.tv_nsec - init.tv_nsec) / 1000000000.0;

	Rcpp::Rcout << "Ready!\n\n";
	
	const ArrayXXd finalse = (exp(ser.array() - 2*log(scale))).block(0, extn, Nsim, t);
	const ArrayXXd finalsg = (exp(sgr.array() - 2*log(scale))).block(0, extn, Nsim, t);

    return Rcpp::List::create(Rcpp::Named("ser") = finalse,
                              Rcpp::Named("sgr") = finalsg,
                              Rcpp::Named("le") = le.array().exp(),
                              Rcpp::Named("lg") = lg.array().exp(),
							  Rcpp::Named("Time_init") = elapsed_init,
							  Rcpp::Named("Time_loop") = elapsed_loop,
                              Rcpp::Named("lasee") = lasee,
                              Rcpp::Named("lasgg") = lasgg);
}


// [[Rcpp::export]]
VectorXd createdata(const MatrixXd& cG, const MatrixXd& cC, const ArrayXd& se, const ArrayXd& sg)
{
	const int t = cC.row(0).size();
	const int n = cG.row(0).size();

	VectorXd y1 = kroneckerProduct(cC, cG)*randomnormal(t*n);

	ArrayXd y11 = y1.array()*sqrt(sg);

	SparseMatrix<double> Id2(n, n);
	Id2.setIdentity();

	VectorXd y2 = kroneckerProduct(cC, Id2)*randomnormal(t*n);

	ArrayXd y22 = y2.array()*sqrt(se);

	return y11 + y22;
}
