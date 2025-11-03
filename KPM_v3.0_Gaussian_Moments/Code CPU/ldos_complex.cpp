#include <iostream>
#include <random>
#include "../../eigen-3.4.0/Eigen/Dense"

#define PI acos(-1)
#define L 4096
#define M 256
#define W 2.5
#define delta 1.0/(2.1 + W/2)
#define mu 1.0/(sqrt(sqrt(2)))
#define sigma sqrt(1.0 - 1.0/sqrt(2))
#define energy 0.0/delta

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_real_distribution<> uniform(-W/2, W/2);
std::uniform_real_distribution<> uniform_fixed(-sqrt(3), sqrt(3));
std::normal_distribution<> normal{0.0, sigma};

// complex version
// -------------------------------------------------

class KPM
{
    private:
    Eigen::ArrayXd a = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd b = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd c = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd r = Eigen::ArrayXd::Zero(L);
    //int M = 181;

    public:
    Eigen::ArrayXd ldos = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd disorder = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd ldos_med_old = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd ldos_med     = Eigen::ArrayXd::Zero(L);
    Eigen::ArrayXd ldos_var     = Eigen::ArrayXd::Zero(L);

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

        for (unsigned i = 1; i < L - 1; i++)
            b[i] = (- (r[i - 1] + r[i + 1]) + disorder[i]*r[i])/delta;

        b[0] = (- r[1] + disorder[0]*r[0])/delta;
        b[L - 1] = (- r[L - 2] + disorder[L - 1]*r[L - 1])/delta;
    }

    double kpm_coefficient(double e, double m)
    {
        double v  = PI/(double(M) + 1);
        double jk = ((M - m + 1)*cos(v*m) + sin(v*m)/tan(v))/double(M + 1);
        double ch = cos(m*acos(e));
        double rt = 1/(PI*sqrt(1 - e*e));

        return jk*ch*rt;
    }

    void iteration()
    {
        ldos += a*kpm_coefficient(energy, 0);
        ldos += 2*b*kpm_coefficient(energy, 1);

        for (unsigned j = 0; j < M - 2; j++)
        {
            if (j%2 == 0)
            {
                for (unsigned i = 1; i < L - 1; i++)
                    a[i] = 2*(- (b[i - 1] + b[i + 1]) + disorder[i]*b[i])/delta - a[i];

                a[0] = 2*(- b[1] + disorder[0]*b[0])/delta - a[0];
                a[L - 1]= 2*(- b[L - 2] + disorder[L - 1]*b[L - 1])/delta - a[L - 1];

                ldos += 2*a*kpm_coefficient(energy, j + 2);
            }

            else
            {
                for (unsigned i = 1; i < L - 1; i++)
                    b[i] = 2*(- (a[i - 1] + a[i + 1]) + disorder[i]*a[i])/delta - b[i];

                b[0] = 2*(- a[1] + disorder[0]*a[0])/delta - b[0];
                b[L - 1]= 2*(- a[L - 2] + disorder[L - 1]*a[L - 1])/delta - b[L - 1];
            
                ldos += 2*b*kpm_coefficient(energy, j + 2);
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
    Eigen::VectorXd subdiagonal = Eigen::VectorXd::Zero(L - 1);
    Eigen::VectorXd eigen_values = Eigen::VectorXd::Zero(L);
    Eigen::MatrixXd eigen_vector = Eigen::MatrixXd::Zero(L, L);
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
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sol;
        sol.computeFromTridiagonal(disorder/delta, subdiagonal/delta);

        eigen_values = sol.eigenvalues();
        eigen_vector = sol.eigenvectors();
    }
    
    double kpm_coefficient(double e, double e_v, double m)
    {
        double v  = PI/(double(M) + 1);
        double jk = ((M - m + 1)*cos(v*m) + sin(v*m)/tan(v))/double(M + 1);
        double ch1 = cos(m*acos(e));
        double ch2 = cos(m*acos(e_v));

        return jk*ch1*ch2;
    }

    double build_peak(double e, double e_v)
    {
        double peak = 0;
        peak += kpm_coefficient(e, e_v, 0);

        for (unsigned i = 1; i < M; i++)
            peak += 2*kpm_coefficient(e, e_v, i);

        peak *= 1/(PI*sqrt(1 - e*e));

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
            double peak = build_peak(e, eigen_values[i]);
            ldos_exact_var += eigen_vector.col(i).array()*eigen_vector.col(i).array()*peak*peak;
        }

        ldos_exact_var /= L*L;
    }
    
    void build_exact_ldos(double e)
    {
        for (unsigned i = 0; i < L; i++)
        {
            //M = 256;
            //double peak = build_peak_gaussian(e, eigen_values[i], PI/double(M));
            double peak = build_peak(e, eigen_values[i]);
            ldos_exact += eigen_vector.col(i).array()*eigen_vector.col(i).array()*peak;
        }

        ldos_exact /= L;
    }
};

// -------------------------------------------------

int main()
{
    for (unsigned runs = 0; runs < 3; runs++)
    {
        Eigen::ArrayXd ldos = Eigen::ArrayXd::Zero(L);
        Eigen::ArrayXd variance = Eigen::ArrayXd::Zero(L);
        int num_medias = 128;

        KPM kpm;
        EXACT exa_var;

        kpm.init_disorder();

        for (unsigned i = 0; i < 1; i++)
        {
            kpm.build_average_ldos(num_medias);
            
            ldos = (kpm.ldos_med - ldos)/(i + 1);
            variance += (kpm.ldos_var - variance)/(i + 1);
        }

        exa_var.disorder = kpm.disorder;
        exa_var.fill_subdiagonal();
        exa_var.diagonalize();
        exa_var.build_exact_var(energy);
        exa_var.build_exact_ldos(energy);

        // ---------------------------------------------

        for (unsigned i = 0; i < L; i++)
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
                          << exp(log(exa_var.ldos_exact[i]) - log(ldos[i])) - 1 << "\n";
    }

    return 1;
}
