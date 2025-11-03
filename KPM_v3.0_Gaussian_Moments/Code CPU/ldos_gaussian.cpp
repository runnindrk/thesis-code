#include <iostream>
#include <fstream>
#include <random>
#include "./eigen-3.4.0/Eigen/Dense"

#define PI acos(-1)
#define L 31768
#define M 2
#define W 2.5
#define delta 1.0/((4.0 + W) - 0.01)
#define mu 1.0/(sqrt(sqrt(2)))
#define sigma sqrt(1.0 - 1.0/sqrt(2))
#define energy 0.0/delta

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_real_distribution<> uniform(-W/2, W/2);
std::uniform_real_distribution<> uniform_fixed(-sqrt(3), sqrt(3));
std::normal_distribution<> normal{0.0, sigma};

// -------------------------------------------------

class KPM
{
    private:

    Eigen::ArrayXd readMoments(int size)
    {   
        Eigen::ArrayXd m = Eigen::ArrayXd::Zero(size);

        std::ifstream File;
        File.open("gaussian_moments.dat");
        std::vector<double>numbers;
        double number;
        while(File >> number) numbers.push_back(number);

        File.close();

        for(int i = 0; i < size; i++) 
            m[i] = numbers[i];

        return m;
    }

    Eigen::ArrayXd a = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd b = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd c = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd r = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd pre_moments = readMoments(M);
    //int M = 181;

    public:
    Eigen::ArrayXd ldos = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd disorder = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd ldos_med_old = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd ldos_med     = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd ldos_var     = Eigen::ArrayXd::Zero(L);

    int mod(int a, int b)
    {
        int r = a%b;
        return r < 0 ? r + b : r;
    }

    void zero()
    {
        a.setZero(); b.setZero(); r.setZero();
        ldos.setZero();
    }

    void init_disorder()
    {
        for (unsigned i = 0; i < L; i++)
            disorder[i] = uniform(rng);
    }

    void init_random_vector()
    {
        for (unsigned i = 0; i < L; i++)
            r[i] = (uniform(rng) < 0)?normal(rng) + mu:normal(rng) - mu;
    }

    void init_kpm_vectors()
    {
        for (unsigned i = 0; i < L; i++)
            a[i] = r[i];

        for (unsigned i = 0; i < L; i++) // was 1 and L - 1 for open boundaries
            b[i] = (- (r[mod(i - 1, L)] + r[mod(i + 1, L)]) + disorder[i]*r[i])/delta;

        //b[0] = (- r[1] + disorder[0]*r[0])/delta;
        //b[L - 1] = (- r[L - 2] + disorder[L - 1]*r[L - 1])/delta;
    }

    double kpm_coefficient(double e, double m)
    {
        double v  = PI/(double(M) + 1);
        double jk = ((M - m + 1)*cos(v*m) + sin(v*m)/tan(v))/double(M + 1);
        //double ch = cos(m*acos(e));
        double rt = 1/(PI*sqrt(1 - e*e));

        return jk*rt;
    }

    void iteration()
    {
        ldos += a*kpm_coefficient(energy, 0)*pre_moments[0];
        ldos += 2*b*kpm_coefficient(energy, 1)*pre_moments[1];

        for (unsigned j = 0; j < M - 2; j++)
        {
            if (j%2 == 0)
            {
                for (unsigned i = 0; i < L; i++)
                    a[i] = 2*(- (b[mod(i - 1, L)] + b[mod(i + 1, L)]) + disorder[i]*b[i])/delta - a[i];

                //a[0] = 2*(- b[1] + disorder[0]*b[0])/delta - a[0];
                //a[L - 1]= 2*(- b[L - 2] + disorder[L - 1]*b[L - 1])/delta - a[L - 1];

                ldos += 2*a*kpm_coefficient(energy, j + 2)*pre_moments[j + 2];
            }

            else
            {
                for (unsigned i = 0; i < L; i++)
                    b[i] = 2*(- (a[mod(i - 1, L)] + a[mod(i + 1, L)]) + disorder[i]*a[i])/delta - b[i];

                //b[0] = 2*(- a[1] + disorder[0]*a[0])/delta - b[0];
                //b[L - 1]= 2*(- a[L - 2] + disorder[L - 1]*a[L - 1])/delta - b[L - 1];
            
                ldos += 2*b*kpm_coefficient(energy, j + 2)*pre_moments[j + 2];
            }
        }
    }

    void build_ldos()
    {
        zero();
        init_random_vector();
        init_kpm_vectors();
        iteration();

        for (unsigned i = 0; i < L; i++)
            ldos[i] = ldos[i]*r[i]/L;
    }

    void build_average_ldos(int num_medias)
    {
        for (unsigned i = 0; i < num_medias; i++)
        {
            build_ldos();

            ldos_med_old = ldos_med;
            ldos_med += (ldos - ldos_med)/(i + 1);
            ldos_var += ((ldos - ldos_med)*(ldos - ldos_med_old) - ldos_var)/(i + 1);
        }

        ldos_var *= num_medias/double(num_medias - 1);
    }
};

// -------------------------------------------------

class EXACT
{
    private:

    Eigen::ArrayXd readMoments(int size)
    {   
        Eigen::ArrayXd m = Eigen::ArrayXd::Zero(size);

        std::ifstream File;
        File.open("gaussian_moments.dat");
        std::vector<double>numbers;
        double number;
        while(File >> number) numbers.push_back(number);

        File.close();

        for(int i = 0; i < size; i++) 
            m[i] = numbers[i];

        return m;
    }

    Eigen::VectorXd subdiagonal = Eigen::VectorXd::Zero(L - 1);
    Eigen::VectorXd eigen_values = Eigen::VectorXd::Zero(L);
    Eigen::VectorXd eigen_vector_single = Eigen::VectorXd::Zero(L);
    Eigen::MatrixXd eigen_vector = Eigen::MatrixXd::Zero(L, L);
    Eigen::ArrayXd pre_moments = readMoments(M);
    //int M = 256;

    public:
    Eigen::ArrayXd ldos_exact_var = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd ldos_exact = Eigen::ArrayXd::Zero(L);
    Eigen::VectorXd disorder = Eigen::VectorXd::Zero(L);

    void fill_subdiagonal()
    {
        for (unsigned i = 0; i < L - 1; i++)
            subdiagonal[i] = -1.0;
    }
    
    void diagonalize()
    {
    	//valores próprios só
    	
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sol;
        sol.computeFromTridiagonal(disorder/delta, subdiagonal/delta, Eigen::EigenvaluesOnly);

        eigen_values = sol.eigenvalues();
        //eigen_vector = sol.eigenvectors();
    }

    
	Eigen::VectorXd vetor_prop(const Eigen::VectorXd& diag,const Eigen::VectorXd& diag_s, const double& tau) 
    {
        // Henrique Code

	  	Eigen::VectorXd d = Eigen::VectorXd::Zero(L);
	  	Eigen::VectorXd delt = Eigen::VectorXd::Zero(L);
	  	Eigen::VectorXd miu = Eigen::VectorXd::Zero(L);
	  	Eigen::VectorXd z = Eigen::VectorXd::Zero(L);

		d(0) = diag(0) - tau;
		delt(L - 1) = diag(L - 1) - tau;
		for (int i = 1; i < L; i++) {
		    d(i) = diag(i) - tau - pow(diag_s(i - 1), 2) / d(i - 1);
		    delt(L - 1 - i) = diag(L - 1 - i) - tau -
		                        pow(diag_s(L - 1 - i), 2) / delt(L - i);
		}
		miu(0) = delt(0);
		int k = 0;
		for (int i = 1; i < L; i++) {
		    miu(i) = miu(i - 1) * delt(i) / d(i - 1);
		    if (abs(miu(i)) < abs(miu(k))) {
		        k = i;
		    }
		}
		z(k) = 1.;
		for (int i = k - 1; i >= 0; i--) {
		    z(i) = -z(i + 1) * diag_s(i) / d(i);
		}
		for (int i = k + 1; i < L; i++) {
		    z(i) = -diag_s(i - 1) * z(i - 1) / delt(i);
		}
		return z / z.norm();
    }
    
    double kpm_coefficient(double e, double e_v, double m)
    {
        double v  = PI/(double(M) + 1);
        double jk = ((M - m + 1)*cos(v*m) + sin(v*m)/tan(v))/double(M + 1);
        //double ch1 = cos(m*acos(e));
        double ch2 = cos(m*acos(e_v));

        return jk*ch2;
    }

    double build_peak(double e, double e_v)
    {
        double peak = 0;
        peak += kpm_coefficient(e, e_v, 0)*pre_moments[0];

        for (unsigned i = 1; i < M; i++)
            peak += 2*kpm_coefficient(e, e_v, i)*pre_moments[i];

        peak *= 1/(PI*sqrt(1 - (e_v)*(e_v)));

        return peak;
    }

    double build_peak_gaussian(double e, double e_v, double resolution)
    {
        return 1.0/((sqrt(2*PI))*resolution)*exp(-0.5*(e - e_v)*(e - e_v)/(resolution*resolution));
    }

   
    void build_exact_var(double e)
    {
        for (unsigned i = 0; i < L; i++)
        {
            //M = 181;
            //double peak = build_peak_gaussian(e, eigen_values[i], PI/double(M)*sqrt(2));

            if (abs(eigen_values[i] - e) < 0.1*500)
            {
                double peak = build_peak(e, eigen_values[i]);
                eigen_vector_single = vetor_prop(disorder/delta, subdiagonal/delta, eigen_values[i]);
                //ldos_exact_var += eigen_vector.col(i).array()*eigen_vector.col(i).array()*peak*peak;
                ldos_exact_var += eigen_vector_single.array()*eigen_vector_single.array()*peak*peak;
            }
        }

        ldos_exact_var /= L*L;
    }
    
    void build_exact_ldos(double e)
    {
        for (unsigned i = 0; i < L; i++)
        {
            //M = 256;
            //double peak = build_peak_gaussian(e, eigen_values[i], PI/double(M));
            
            if (abs(eigen_values[i] - e) < 0.1*500)
            {
                double peak = build_peak(e, eigen_values[i]);
                eigen_vector_single = vetor_prop(disorder/delta, subdiagonal/delta, eigen_values[i]);
                //ldos_exact += eigen_vector.col(i).array()*eigen_vector.col(i).array()*peak;
                ldos_exact += eigen_vector_single.array()*eigen_vector_single.array()*peak;
            }
        }

        ldos_exact /= L;
    }
};

// -------------------------------------------------

// -------------------------------------------------

Eigen::ArrayXd readMoments(int size)
{   
    Eigen::ArrayXd m = Eigen::ArrayXd::Zero(size);

    std::ifstream File;
    File.open("gaussian_moments.dat");
    std::vector<double>numbers;
    double number;
    while(File >> number) numbers.push_back(number);

    File.close();

    for(int i = 0; i < size; i++) 
        m[i] = numbers[i];

    return m;
}

Eigen::ArrayXd readDisorder(int size)
{   
    Eigen::ArrayXd m = Eigen::ArrayXd::Zero(size);

    std::ifstream File;
    File.open("disorder.dat");
    std::vector<double>numbers;
    double number;
    while(File >> number) numbers.push_back(number);

    File.close();

    for(int i = 0; i < size; i++) 
        m[i] = numbers[i];

    return m;
}

int main()
{   
	Eigen::ArrayXd pre_moments = readMoments(M);
    Eigen::ArrayXd disorder    = readDisorder(L);

	readMoments(M);
    readDisorder(L);
	
	//for (unsigned i = 0; i < M; i++)
    //	std::cout <<  pre_moments[i] << "\n";
    
    
    for (unsigned runs = 0; runs < 1; runs++)
    {
        Eigen::ArrayXd ldos = Eigen::ArrayXd::Zero(L);
        Eigen::ArrayXd variance = Eigen::ArrayXd::Zero(L);
        int num_medias = 128;

        KPM kpm;
        EXACT exa_var;

        kpm.disorder = disorder;
        //kpm.init_disorder();

        for (unsigned i = 0; i < 1; i++)
        {
            kpm.build_average_ldos(num_medias);
            
            ldos = (kpm.ldos_med - ldos)/(i + 1);
            variance += (kpm.ldos_var - variance)/(i + 1);
        }

        //exa_var.disorder = kpm.disorder;
        //exa_var.fill_subdiagonal();
        //exa_var.diagonalize();
        //exa_var.build_exact_var(energy);
        //exa_var.build_exact_ldos(energy);

        // ---------------------------------------------
        for (unsigned i = 0; i < L; i++)
             std::cout << variance[i] << "\n";
        /*for (unsigned i = 0; i < L; i++)
                std::cout << i + runs*L << "  " \
                          << exa_var.ldos_exact_var[i] << " " \ 
                          << variance[i] << "  " \ 
                          << log(exa_var.ldos_exact_var[i]) << " " \ 
                          << log(variance[i]) << " " \
                          << exp(log(exa_var.ldos_exact_var[i]) - log(variance[i])) - 1 << " " \
                          << exa_var.ldos_exact[i] << " " \
                          << ldos[i] << "  " \ 
                          << log(exa_var.ldos_exact[i]) << " " \ 
                          << log(ldos[i]) << " " \
                          << exp(log(exa_var.ldos_exact[i]) - log(ldos[i])) - 1 << "\n";*/
    }

    return 1;
}
