#define BOOST_TEST_MODULE test_pack

#include <iostream>
#include <cmath>

// BOOST
#include <boost/format.hpp>
using fmt = boost::format;
#include <boost/test/included/unit_test.hpp>

// Switch on some debugging code that keeps
// track of the generation of random numbers.
#define MC_ACCUMULATE_RANDOM
#define MC_SAVE_RANDOM

// Switch on fixed seed so we get the same pseudorandom
// sequence every time
#define MC_REPRODUCE

#include "log.hh"       // Set up simple logging
#include "machine.hh"   // Set up and wrap pseudorandom numbers

// Split an array between MPI processes,
// and implement arithmetic
#include "pack.hh"

// Measure performance
#include "performance.hh"

// Close to zero for tests
#define MC_TEST_EPSILON 1.0e-6

// Maximum power to check
#define MC_TEST_MAX_POWER 4

// Maximum power of ten for beta
#define MC_TEST_MAX_BETA_POWER 1

// Scalar used to test arithmetic
#define MC_TEST_SCALAR 2.3

// Repeat numerical operations for timing statistics
#define MC_TEST_REPEATS 1

// Width of string output field
#define MC_TEST_FIELD_STRING "%30s"

// Set up performance timing for numerical sections of the code
timer perf("timing.tsv");

// Pseudorandom number generators
machine<float,
        std::mt19937_64,
        std::normal_distribution<float>> nrm(0.0, 1.0);

// Random vector for testing
random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> u(MC_TEST_SIZE, nrm);

// Format report of two numbers and their difference
void numerical_report(const double x, const double y)
{
    MC_INFO << fmt("%16.8e") % x;
    MC_INFO << fmt("%16.8e") % y;
    MC_INFO << fmt("%16.8e") % (x-y) << std::endl;
}

// Format report of a counter, two numbers, and their difference
void numerical_report(const int k, const double x, const double y)
{
    MC_INFO << fmt("%3d") % k << fmt("%16.8e") % x;
    MC_INFO << fmt("%16.8e") % y;
    MC_INFO << fmt("%16.8e") % (x-y) << std::endl;
}

// Test the non-MPI version of the dimensions class
// BOOST_AUTO_TEST_CASE(test_0_dimensions)
// {
//     std::vector<size_t> m = {2, 2, 2};
//     dimensions M(m);
//     BOOST_CHECK(M.sz == 8);
// }

BOOST_AUTO_TEST_CASE(test_1_random_pack_mean)
{
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> w(MC_TEST_SIZE, nrm);
    w[0] /= 0.0; // Make a NaN to test the NaN catcher.
    double mn = 0.0;
    double wmn = 0.0;

    // Time the pack<..>.mean() method
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
        wmn = w.mean();
    perf.finish_line(MC_TEST_REPEATS, "test_1_random_pack_mean");

    // Note that we're skipping the INF in position 0
    for(size_t j=1; j<w.sz; ++j)
        mn += w[j];
    mn /= (w.sz-1);

    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "mean: ";
    numerical_report(mn, wmn);
    BOOST_CHECK(abs(wmn - mn) < MC_TEST_EPSILON);
    //BOOST_CHECK(wmn == mn);
}

BOOST_AUTO_TEST_CASE(test_2_random_pack_mean_squared)
{
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> w(MC_TEST_SIZE, nrm);
    w[0] /= 0.0; // Make a NaN to test the NaN catcher.
    double mn2 = 0.0;
    double wmn2 = 0.0;

    // Time the pack<..>.mean_square() method
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
        wmn2 = w.mean_square();
    perf.finish_line(MC_TEST_REPEATS, "test_2_random_pack_mean_squared");

    // Note that we're skipping the INF in position 0
    for(size_t j=1; j<w.sz; ++j)
        mn2 += w[j] * w[j];
    mn2 /= (w.sz-1);

    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "mean-square: ";
    numerical_report(mn2, wmn2);
    BOOST_CHECK(abs(wmn2 - mn2) < MC_TEST_EPSILON);
    // BOOST_CHECK(wmn2 == mn2);
}

BOOST_AUTO_TEST_CASE(test_3_random_pack_variance)
{
    float tmp;
    double umn = 0.0;
    double umn2 = 0.0;    

    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        umn = u.mean();
        umn2 = u.mean_square();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_3_random_pack_variance");

    // Short cut to variance from mean and mean-square
    double uvariance = umn2 - umn*umn;

    // Obtain the variance directly
    double variance = 0.0;
    for(size_t j=0; j<u.sz; ++j)
    {
        tmp = u[j] - umn;
        variance += tmp * tmp;
    }

    // Use 1/N, not 1/(N-1), because we're just testing against
    // the result from the mean() and mean_square() methods.
    variance /= u.sz;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "variance: ";
    numerical_report(variance, uvariance);
    BOOST_CHECK(abs(uvariance - variance) < MC_TEST_EPSILON);
    // BOOST_CHECK(uvariance == variance);
}

BOOST_AUTO_TEST_CASE(test_4_random_pack_overload_plus_unary_minus)
{
    double mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(-x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);

    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "move=, plus: ";

    // Check move assignment after addition, using mean() method
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x + y;
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_4_random_pack...plus_unary_minus");

    MC_INFO << fmt("%16.8e") % mn << std::endl;
    BOOST_CHECK(abs(mn) <= MC_TEST_EPSILON);
    BOOST_CHECK(mn == 0.0);
}

BOOST_AUTO_TEST_CASE(test_5_random_pack_overload_increment)
{
    double mn = 0.0;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "increment: ";

    // Use mean to check increment and decrement,
    // with subtraction and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = (++x) - (--y);
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_5_random_pack_overload_increment");

    MC_INFO << fmt("%16.8e") % abs(mn - 2.0) << std::endl;
    BOOST_CHECK(abs(mn - 2.0) <= MC_TEST_EPSILON);
    // BOOST_CHECK(mn == 2.0);
}


BOOST_AUTO_TEST_CASE(test_6_random_pack_overload_pack_minus)
{
    double mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    pack<float, double, float> z(MC_TEST_SIZE);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "pack minus: ";

    // Use mean() to check subtraction and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x - y;
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_6_random_pack...pack_minus");

    MC_INFO << boost::format("%16.8e") % mn << std::endl;
    BOOST_CHECK(abs(mn) >= 0.0);
    BOOST_CHECK(abs(mn) <= MC_TEST_EPSILON);
    BOOST_CHECK(mn == 0.0);
}

BOOST_AUTO_TEST_CASE(test_7_random_pack_overload_minus1)
{
    double mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "minus: ";

    // Check random_pack move assignment after subtraction,
    // susing mean() method
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x - y;
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_7_random_pack_overload_minus1");

    MC_INFO << fmt("%16.8e") % mn << std::endl;
    BOOST_CHECK(abs(mn) <= MC_TEST_EPSILON);
    BOOST_CHECK(mn == 0.0);
}

BOOST_AUTO_TEST_CASE(test_8_random_pack_overload_minus2)
{
    double xmn, mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "scalar minus: ";

    // Use mean() to check scalar subtraction and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        xmn = x.mean();
        z = x - MC_TEST_SCALAR;
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_8_random_pack_overload_minus2");

    numerical_report(xmn - MC_TEST_SCALAR, mn);
    BOOST_CHECK(abs(mn - xmn + MC_TEST_SCALAR) <= MC_TEST_EPSILON);
    // BOOST_CHECK(mn - xmn == MC_TEST_SCALAR);
}

BOOST_AUTO_TEST_CASE(test_9_random_pack_overload_plus2)
{
    double xmn, mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "scalar plus: ";

    // Use mean() to check scalar addition and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        xmn = x.mean();
        z = x + MC_TEST_SCALAR;
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_9_random_pack_overload_plus2");

    numerical_report(xmn + MC_TEST_SCALAR, mn);
    BOOST_CHECK(abs(mn - xmn - MC_TEST_SCALAR) <= MC_TEST_EPSILON);
    // BOOST_CHECK(mn - xmn == MC_TEST_SCALAR);
}

BOOST_AUTO_TEST_CASE(test_10_random_pack_overload_multiply1)
{
    double mn = 0.0;
    double zmn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "multiply: ";

    // Use mean() to check multiplication and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x * y;
        zmn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_10_random_pack_overload_multiply1");

    // Repeat calculation with direct access
    for(size_t j=0; j<x.sz; ++j)
        mn += x[j] * y[j];
    mn /= x.sz;

    numerical_report(zmn, mn);
    BOOST_CHECK(abs(zmn - mn) <= MC_TEST_EPSILON);
    //BOOST_CHECK(mn == zmn);
}

BOOST_AUTO_TEST_CASE(test_11_random_pack_overload_multiply2)
{
    double xmn, mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "scalar multiply: ";

    // Use mean() to check scalar multiplication and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        xmn = x.mean();
        z = x * MC_TEST_SCALAR;
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_11_random_pack_overload_multiply2");

    numerical_report(xmn * MC_TEST_SCALAR, mn);
    BOOST_CHECK(abs(mn - xmn * MC_TEST_SCALAR) <= MC_TEST_EPSILON);
    // BOOST_CHECK(mn - xmn == MC_TEST_SCALAR);
}

BOOST_AUTO_TEST_CASE(test_12_random_pack_overload_convolve)
{
    double mn = 0.0;
    double zmn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "convolve: ";

    // Use mean() to check 'convolution' and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x % y;
        zmn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_12_random_pack_overload_convolve");

    // Repeat calculation with direct access
    for(size_t j=0; j<x.sz; ++j)
        mn += x[j] * y[j];
    mn /= x.sz;

    numerical_report(zmn, mn);
    BOOST_CHECK(abs(zmn - mn) <= MC_TEST_EPSILON);
    //BOOST_CHECK(mn == zmn);
}

BOOST_AUTO_TEST_CASE(test_13_random_pack_overload_divide1)
{
    size_t count = 0;
    double mn = 0.0;
    double zmn = 0.0;
    double tmp;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "elementwise divide: ";

    // Use mean() to check division and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x / y;
        zmn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_13_random_pack_overload_divide1");

    // Repeat calculation with direct access
    // Note that here we exclude NaNs and INFs using tmp==tmp,
    // but in pack<..> methods std::isfinite() is used.
    for(size_t j=0; j<x.sz; ++j)
    {
        tmp = x[j] / y[j];
        if(tmp==tmp)
        {
            count++;
            mn += tmp;
        }
    }
    mn /= count;

    numerical_report(zmn, mn);
    BOOST_CHECK(abs(zmn - 1.0) <= MC_TEST_EPSILON);
    BOOST_CHECK(abs(zmn - mn) <= MC_TEST_EPSILON);
    BOOST_CHECK(mn == zmn);
}

BOOST_AUTO_TEST_CASE(test_14_random_pack_overload_divide2)
{
    double xmn, mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "scalar divide: ";

    // Use mean() to check division and move assignment
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        xmn = x.mean();
        z = x / MC_TEST_SCALAR;
        mn = z.mean();
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_14_random_pack_overload_divide2");

    numerical_report(xmn / MC_TEST_SCALAR, mn);
    BOOST_CHECK(abs(mn - xmn / MC_TEST_SCALAR) <= MC_TEST_EPSILON);
    // BOOST_CHECK(mn - xmn == MC_TEST_SCALAR);
}

BOOST_AUTO_TEST_CASE(test_15_random_pack_overload_assign_plus)
{
    double mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(-x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "assign plus: ";

    // Use mean() to check assign and add
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x;
        z += y;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_15_random_pack_overload_assign_plus");
    mn = z.mean();
    MC_INFO << fmt("%16.8e") % mn << std::endl;
    BOOST_CHECK(abs(mn) <= MC_TEST_EPSILON);
    BOOST_CHECK(mn == 0.0);
}

BOOST_AUTO_TEST_CASE(test_16_random_pack_overload_assign_minus)
{
    double mn;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "assign minus: ";
    
    // Use mean() to check assign and subtract
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x;
        z -= y;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_16_random_pack_overload_assign_minus");
    mn = z.mean();
    MC_INFO << fmt("%16.8e") % mn << std::endl;
    BOOST_CHECK(abs(mn) <= MC_TEST_EPSILON);
    BOOST_CHECK(mn == 0.0);
}

BOOST_AUTO_TEST_CASE(test_17_random_pack_overload_assign_times)
{
    double mn = 0.0;
    double zmn = 0.0;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "assign times: ";

    // Use mean to check multiply-assign
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x;
        z *= y;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_17_random_pack_overload_assign_times");

    // Check by direct multiplication
    zmn = z.mean();
    for(size_t j=0; j<x.sz; ++j)
        mn += x[j] * y[j];
    mn /= x.sz;

    numerical_report(zmn, mn);
    BOOST_CHECK(abs(zmn - mn) <= MC_TEST_EPSILON);
    //BOOST_CHECK(mn == zmn);
}

BOOST_AUTO_TEST_CASE(test_18_random_pack_overload_assign_divide)
{
    size_t count = 0;
    double tmp;
    double mn = 0.0;
    double zmn = 0.0;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(x);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO <<  fmt(MC_TEST_FIELD_STRING) % "assign divide: ";

    // Use mean to check divide-assign
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        z = x;
        z /= y;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_18_random_pack_overload_assign_divide");
    
    // Check division using tmp==tmp to
    // omit the NaNs and INFs
    zmn = z.mean();
    for(size_t j=0; j<x.sz; ++j)
    {
        tmp = x[j] / y[j];
        if(tmp == tmp)
        {
            ++count;
            mn += tmp;
        }
    }
    mn /= count;

    numerical_report(zmn, mn);
    BOOST_CHECK(abs(zmn - 1.0) <= MC_TEST_EPSILON);
    BOOST_CHECK(abs(zmn - mn) <= MC_TEST_EPSILON);
    BOOST_CHECK(mn == zmn);
}

BOOST_AUTO_TEST_CASE(test_19_pack_overload_test_nullptr_copy)
{
    double sum = 0.0;
    pack<float, double, float> x(MC_TEST_SIZE);
    pack<float, double, float> y(MC_TEST_SIZE);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check copy assign: ";
    // Set x.x to nullptr to check
    // handling of nullptr by copy construction 
    delete [] x.x;
    x.x = nullptr;
    // Set constant y
    y = MC_TEST_SCALAR;
    x = y;
    // Check copy assignment by rms
    for(size_t j=0; j<x.sz; ++j)
        sum += (x[j] - y[j]) * (x[j] - y[j]);
    sum /= x.sz;
    sum = sqrt(sum);
    MC_INFO << fmt("%16.8e") % sum << std::endl;
    BOOST_CHECK(sum < MC_TEST_EPSILON);
}

// BOOST_AUTO_TEST_CASE(test_20_pack_exception_on_length_mismatch)
// {
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
//     pack<float, double, float> x(MC_TEST_SIZE+1);
//     pack<float, double, float> y(MC_TEST_SIZE);
//     BOOST_CHECK_THROW(x = y, std::domain_error);
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
// }

// BOOST_AUTO_TEST_CASE(test_21_pack_exception_on_length_mismatch)
// {
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
//     pack<float, double, float> x(MC_TEST_SIZE+1);
//     pack<float, double, float> y(MC_TEST_SIZE+1);
//     pack<float, double, float> z(MC_TEST_SIZE);
//     BOOST_CHECK_THROW(z = x + y, std::domain_error);
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
// }

BOOST_AUTO_TEST_CASE(test_22_pack_exception_out_of_bounds)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
    pack<float, double, float> x(MC_TEST_SIZE);
    BOOST_CHECK_THROW(x[-1], std::out_of_range);
    BOOST_CHECK_THROW(x[MC_TEST_SIZE], std::out_of_range);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
}

// BOOST_AUTO_TEST_CASE(test_23_pack_exception_mismatch_plus)
// {
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
//     pack<float, double, float> x(MC_TEST_SIZE+1);
//     pack<float, double, float> y(MC_TEST_SIZE);
//     BOOST_CHECK_THROW(x + y, std::domain_error);
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
// }

// BOOST_AUTO_TEST_CASE(test_24_pack_exception_mismatch_minus)
// {
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
//     pack<float, double, float> x(MC_TEST_SIZE+1);
//     pack<float, double, float> y(MC_TEST_SIZE);
//     BOOST_CHECK_THROW(x - y, std::domain_error);
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
// }

BOOST_AUTO_TEST_CASE(test_25_pack_exception_mismatch_times)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
    pack<float, double, float> x(MC_TEST_SIZE+1);
    pack<float, double, float> y(MC_TEST_SIZE);
    BOOST_CHECK_THROW(x * y, std::domain_error);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
}

BOOST_AUTO_TEST_CASE(test_26_pack_exception_mismatch_divide)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
    pack<float, double, float> x(MC_TEST_SIZE+1);
    pack<float, double, float> y(MC_TEST_SIZE);
    BOOST_CHECK_THROW(x / y, std::domain_error);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
}

BOOST_AUTO_TEST_CASE(test_27_pack_exception_mismatch_convolve)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
    pack<float, double, float> x(MC_TEST_SIZE+1);
    pack<float, double, float> y(MC_TEST_SIZE);
    BOOST_CHECK_THROW(x % y, std::domain_error);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
}

BOOST_AUTO_TEST_CASE(test_28_pack_self_assignment)
{
    double sum = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check self assign: ";
    pack<float, double, float> x(MC_TEST_SIZE);
    pack<float, double, float> y(MC_TEST_SIZE);
    // Make sure self assignment has no effect
    x = MC_TEST_SCALAR;
    y = x;
    x = x;
    // RMS difference to check if x is unchanged
    // since it was saved in y
    for(size_t j=0; j<x.sz; ++j)
        sum += (x[j] - y[j]) * (x[j] - y[j]);
    sum /= x.sz;
    sum = sqrt(sum);
    BOOST_CHECK(sum <= MC_TEST_EPSILON);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
}

BOOST_AUTO_TEST_CASE(test_29_pack_overload_test_nullptr_move)
{
    double sum = 0.0;
    pack<float, double, float> x(MC_TEST_SIZE);
    pack<float, double, float> y(MC_TEST_SIZE);
    pack<float, double, float> z(MC_TEST_SIZE);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check move1 assign: ";
    // Set up pack object with missing array
    // and x,x a nullptr
    delete [] x.x;
    x.x = nullptr;
    y = MC_TEST_SCALAR;
    z = 2*MC_TEST_SCALAR;
    // Check move assignment
    x = z - y;
    // RMS difference to check
    for(size_t j=0; j<x.sz; ++j)
        sum += (x[j] - y[j]) * (x[j] - y[j]);
    sum /= x.sz;
    sum = sqrt(sum);
    MC_INFO << fmt("%16.8e") % sum << std::endl;
    BOOST_CHECK(sum < MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_30_random_pack_overload_test_nullptr_move)
{
    double sum = 0.0;
    double ymn = 0.0;
    double zmn = 0.0;
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> y(MC_TEST_SIZE, nrm);
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> z(MC_TEST_SIZE, nrm);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check move2 assign: ";
    // Checking move assignment with a nullptr
    // and a missing  
    delete [] x.x;
    x.x = nullptr;
    for(size_t j=0; j<x.sz; ++j)
    {
        y[j] = MC_TEST_SCALAR;
        z[j] = MC_TEST_SCALAR;
    }
    x = z - y;
    // Check that y and z have the correct mean,
    // and x is zero after subtraction
    for(size_t j=0; j<x.sz; ++j)
    {
        sum += x[j] * x[j];
        ymn += y[j];
        zmn += z[j];
    }
    sum /= x.sz;
    ymn /= y.sz;
    zmn /= z.sz;
    sum = sqrt(sum);
    MC_INFO << fmt("%16.8e") % sum << std::endl;
    BOOST_CHECK(sum < MC_TEST_EPSILON);
    BOOST_CHECK(fabs(ymn - MC_TEST_SCALAR) < MC_TEST_EPSILON);
    BOOST_CHECK(fabs(zmn - MC_TEST_SCALAR) < MC_TEST_EPSILON);
}

// BOOST_AUTO_TEST_CASE(test_31_pack_assign_to_null)
// {
//     pack<float, double, float> x(MC_TEST_SIZE);
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
//     delete [] x.x;
//     x.x = nullptr;
//     BOOST_CHECK_THROW(x = MC_TEST_SCALAR, std::length_error);
//     MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
// }

BOOST_AUTO_TEST_CASE(test_32_random_pack_mean_allinf)
{
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> w(MC_TEST_SIZE, nrm);
    for(size_t j=0; j<w.sz; ++j)
        w[j] /= 0.0; // Make NaNs to test the NaN catcher.
    double mn = 0.0;
    double wmn = w.mean();
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "mean: ";
    numerical_report(wmn, mn);
    BOOST_CHECK(abs(wmn - mn) < MC_TEST_EPSILON);
    BOOST_CHECK(wmn == mn);
}

BOOST_AUTO_TEST_CASE(test_33_random_pack_mean_squared_allinf)
{
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> w(MC_TEST_SIZE, nrm);
    for(size_t j=0; j<w.sz; ++j)
        w[j] /= 0.0; // Make NaNs to test the NaN catcher.
    double mn2 = 0.0, wmn2 = w.mean_square();
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "squared mean: ";
    numerical_report(wmn2, mn2);
    BOOST_CHECK(abs(wmn2 - mn2) < MC_TEST_EPSILON);
    BOOST_CHECK(wmn2 == 0.0);
    BOOST_CHECK(mn2 == 0.0);
}

BOOST_AUTO_TEST_CASE(test_34_pack_overload_assign_scalar)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "scalar assign: ";
    pack<float, double, float> x(MC_TEST_SIZE);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = 1.0;
    }
    perf.finish_line(MC_TEST_REPEATS, "test_34_pack_overload_assign_scalar");
    double mn = x.mean();
    MC_INFO << fmt("%16.8e") % abs(mn - 1.0) << std::endl;
    BOOST_CHECK(abs(mn - 1.0) <= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_35_pack_overload_move_assign_scalar)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "scalar assign: ";
    double a = 0.3, b = 0.7;
    pack<float, double, float> x(MC_TEST_SIZE);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = a + b;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_35_pack_overload_move assign_scalar");
    double mn = x.mean();
    MC_INFO << boost::format("%16.8e") % abs(mn - 1.0) << std::endl;
    BOOST_CHECK(abs(mn - 1.0) <= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_36_random_pack_overload_assign_scalar)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "scalar assign: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = 1.0;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_36_random_pack_overload_assign_scalar");
    double mn = x.mean();
    MC_INFO << boost::format("%16.8e") % abs(mn - 1.0) << std::endl;
    BOOST_CHECK(abs(mn - 1.0) <= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_37_random_pack_overload_assign_scalar_null)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "check throw: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    delete [] x.x;
    x.x = nullptr;
    BOOST_CHECK_THROW(x = 1.0, std::length_error);
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "done" << std::endl;
}

BOOST_AUTO_TEST_CASE(test_38_pack_overload_assign_scalar_plus)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "scalar assign: ";
    pack<float, double, float> x(MC_TEST_SIZE);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = 0.0;
        x += 1.0;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_38_pack_overload_assign_scalar_plus");
    double mn = x.mean();
    MC_INFO << boost::format("%16.8e") % abs(mn - 1.0) << std::endl;
    BOOST_CHECK(abs(mn - 1.0) <= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_39_pack_overload_assign_scalar_minus)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "scalar assign: ";
    pack<float, double, float> x(MC_TEST_SIZE);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = 0.0;
        x -= 1.0;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_39_pack_overload_assign_scalar_minus");
    double mn = x.mean();
    MC_INFO << fmt("%16.8e") % abs(mn + 1.0) << std::endl;
    BOOST_CHECK(abs(mn + 1.0) <= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_40_pack_overload_assign_scalar_times)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "scalar assign: ";
    pack<float, double, float> x(MC_TEST_SIZE);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = 0.5;
        x *= 2.0;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_40_pack_overload_assign_scalar_times");
    double mn = x.mean();
    MC_INFO << fmt("%16.8e") % abs(mn - 1.0) << std::endl;
    BOOST_CHECK(abs(mn - 1.0) <= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_41_pack_overload_assign_scalar_divide)
{
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "scalar assign: ";
    pack<float, double, float> x(MC_TEST_SIZE);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = 2.0;
        x /= 2.0;
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_41_pack_overload_assign_scalar_divide");
    double mn = x.mean();
    MC_INFO << fmt("%16.8e") % abs(mn - 1.0) << std::endl;
    BOOST_CHECK(abs(mn - 1.0) <= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_42_random_pack_abs)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "abs: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = y;
        x.abs();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_42_random_pack_abs");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j]-abs(y[j]))*(x[j]-abs(y[j]));
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}


BOOST_AUTO_TEST_CASE(test_43_random_pack_square)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "square: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = y;
        x.square();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_43_random_pack_square");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j]-y[j]*y[j])*(x[j]-y[j]*y[j]);
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_44_random_pack_reciprocal)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "reciprocal: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = y;
        x.reciprocal();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_44_random_pack_reciprocal");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j] - 1/y[j])*(x[j] - 1/y[j]);
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_45_random_pack_power)
{
    double rmserr = 0.0;
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    for(int k=0; k<MC_TEST_MAX_POWER; ++k)
    {
        MC_INFO << fmt(MC_TEST_FIELD_STRING) % "power: " << fmt("%3d") % k;
        std::stringstream ss;
        ss << "test_45_random_pack_power_" << k;
        std::string s = ss.str();
        rmserr = 0.0;
        perf.start_timer();
        for(int kk=0;kk<MC_TEST_REPEATS;++kk)
        {
            x = y;
            x.power(k);
        }
        perf.finish_line(MC_TEST_REPEATS, s);
        for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
            rmserr += (x[j] - pow(y[j],k))*(x[j] - pow(y[j],k));
        rmserr /= x.sz;
        rmserr = sqrt(rmserr);
        MC_INFO << fmt("%16.8e") % rmserr << std::endl;
        BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
        x = y;
    }
}

BOOST_AUTO_TEST_CASE(test_46_random_pack_power_square)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "power square: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        x.square();
        y.power(2);
    }
    perf.finish_line(MC_TEST_REPEATS, "test_46_random_pack_power_square");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j] - y[j])*(x[j] - y[j]);
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_47_random_pack_power_fourth)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "power fourth: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        x.square();
        x.square();
        y.power(4);
    }
    perf.finish_line(MC_TEST_REPEATS, "test_47_random_pack_power_fourth");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j] - y[j])*(x[j] - y[j]);
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_48_random_pack_exp)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "exp: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {   
        x = y;
        x.exp();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_48_random_pack_exp");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j] - exp(y[j]))*(x[j] - exp(y[j]));
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_49_random_pack_mayerf)
{
    double rmserr;
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    int kmax = 2*MC_TEST_MAX_BETA_POWER;
    double beta, delta;
    for(int k=-kmax; k<=0; ++k)
    {
        beta = pow(10.0, 0.5*k);
        rmserr = 0.0;
        std::stringstream ss;
        MC_INFO << fmt(MC_TEST_FIELD_STRING) % "mayerf: ";
        MC_INFO << fmt("%16.8e") % beta;
        ss << "test_49_random_pack_mayerf" << fmt("%16.8e") % beta;
        std::string s = ss.str();
        perf.start_timer();
        for(int k=0;k<MC_TEST_REPEATS;++k)
        {
            x = y;
            x.mayerf(beta);
        }
        perf.finish_line(MC_TEST_REPEATS, s);
        for(size_t j=0; j<x.sz; ++j)
        {
            delta = x[j] - (std::exp(-beta*y[j])-1.0);
            rmserr += delta * delta;
        }
        rmserr /= x.sz;
        rmserr = sqrt(rmserr);
        MC_INFO << fmt("%16.8e") % rmserr << std::endl;
        BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
    }
}

BOOST_AUTO_TEST_CASE(test_50_random_pack_mayerf2)
{
    double rmserr;
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    int kmax = MC_TEST_MAX_BETA_POWER;
    double beta, delta;
    for(int k=-kmax; k<=0; ++k)
    {
        rmserr = 0.0;
        beta = pow(10.0, k);
        std::stringstream ss;
        MC_INFO << fmt(MC_TEST_FIELD_STRING) % " mayerf2: ";
        MC_INFO << fmt("%16.8e") % beta;
        ss << "test_50_random_pack_mayerf2" << fmt("%16.8e") % beta;
        std::string s = ss.str();
        perf.start_timer();
        for(int k=0;k<MC_TEST_REPEATS;++k)
        {
            x = y;
            z = y;
            x.mayerf(beta);
            z *= -beta;
            z.exp();
            z -= 1.0;
        }
        perf.finish_line(MC_TEST_REPEATS, s);
        for(size_t j=0; j<x.sz; ++j)
        {
            delta = x[j] - z[j];
            rmserr += delta * delta;
        }
        rmserr /= x.sz;
        rmserr = sqrt(rmserr);
        MC_INFO << fmt("%16.8e") % rmserr << std::endl;
        BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
    }
}

float testexp(float x)
{
    return exp(x);
}

BOOST_AUTO_TEST_CASE(test_51_random_pack_apply_exp)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "exp: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        x.apply(&testexp);
        y.exp();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_51_random_pack_apply_exp");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j] - y[j])*(x[j] - y[j]);
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}

float testsquare(float x)
{
    return x*x;
}

BOOST_AUTO_TEST_CASE(test_52_random_pack_apply_square)
{
    double rmserr = 0.0;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "apply square: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        x.apply(&testsquare);
        y.square();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_52_random_pack_apply_square");
    for(size_t j=0; j<x.sz; ++j) // Note no check for NaN or INF.
        rmserr += (x[j] - y[j])*(x[j] - y[j]);
    rmserr /= x.sz;
    rmserr = sqrt(rmserr);
    MC_INFO << fmt("%16.8e") % rmserr << std::endl;
    BOOST_CHECK(rmserr<= MC_TEST_EPSILON);
}

BOOST_AUTO_TEST_CASE(test_53_random_pack_rms)
{
    double rmsx, rmsy;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "rms: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        x.square();
        rmsx = x.mean();
        rmsx = sqrt(rmsx);
        rmsy = y.rms();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_53_random_pack_rms");
    numerical_report(rmsx, rmsy);
    BOOST_CHECK(abs(rmsx-rmsy)<= MC_TEST_EPSILON);
    BOOST_CHECK(rmsx == rmsy);
}

BOOST_AUTO_TEST_CASE(test_54_random_pack_meanabs)
{
    double meanabsx, meanabsy;
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "meanabs: ";
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        x.abs();
        meanabsx = x.mean();
        meanabsy = y.mean_abs();
    }
    perf.finish_line(MC_TEST_REPEATS, "test_54_random_pack_meanabs");
    numerical_report(meanabsx, meanabsy);
    BOOST_CHECK(abs(meanabsx - meanabsy)<= MC_TEST_EPSILON);
    BOOST_CHECK(meanabsx == meanabsy);
}

BOOST_AUTO_TEST_CASE(test_55_random_pack_moments)
{
    std::vector<double> a, b;
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        a.push_back(1.0);
        a.push_back(x.mean());
        a.push_back(x.mean_square());
        x = x * z * z;
        a.push_back(x.mean());
        x = x * z;
        a.push_back(x.mean());
        b = y.moments(4);
    }
    perf.finish_line(MC_TEST_REPEATS, "test_55_random_pack_moments");
    for(int k=0;k<=4;++k)
    {
        MC_INFO << fmt(MC_TEST_FIELD_STRING) % "moments: ";
        numerical_report(k, a[k], b[k]);
        BOOST_CHECK(abs(a[k] - b[k])<= MC_TEST_EPSILON);
        // BOOST_CHECK(a[k] == b[k]);
    }
}

BOOST_AUTO_TEST_CASE(test_56_random_pack_mean_abs_allinf)
{
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> w(MC_TEST_SIZE, nrm);
    for(size_t j=0; j<w.sz; ++j)
        w[j] /= 0.0; // Make NaNs to test the NaN catcher.
    double mnabs = 0.0, wmnabs = w.mean_abs();
    MC_INFO << fmt(MC_TEST_FIELD_STRING) % "absolute mean: ";
    numerical_report(wmnabs, mnabs);
    BOOST_CHECK(abs(wmnabs - mnabs) < MC_TEST_EPSILON);
    BOOST_CHECK(wmnabs == mnabs);
}

BOOST_AUTO_TEST_CASE(test_57_random_pack_moments_inf)
{
    std::vector<double> a, b;
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    z[0] /= 0.0;
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        a.push_back(1.0);
        a.push_back(x.mean());
        a.push_back(x.mean_square());
        x = x * z * z;
        a.push_back(x.mean());
        x = x * z;
        a.push_back(x.mean());
        b = y.moments(4);
    }
    perf.finish_line(MC_TEST_REPEATS, "test_57_random_pack_moments_inf");
    for(int k=0;k<=4;++k)
    {
        MC_INFO << fmt(MC_TEST_FIELD_STRING) % "moments with inf: ";
        numerical_report(k, a[k], b[k]);
        BOOST_CHECK(abs(a[k] - b[k])<= MC_TEST_EPSILON);
        // BOOST_CHECK(a[k] == b[k]);
    }
}


BOOST_AUTO_TEST_CASE(test_58_random_pack_moments_allinf)
{
    std::vector<double> a, b;
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> x(MC_TEST_SIZE, nrm);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> y(x);
    random_pack<float,
            double,
            float,
            std::mt19937_64,
            std::normal_distribution<float>> z(x);
    perf.start_timer();
    z /= 0.0;
    for(int k=0;k<MC_TEST_REPEATS;++k)
    {
        x = z;
        y = z;
        a.push_back(0.0);
        a.push_back(x.mean());
        a.push_back(x.mean_square());
        x = x * z * z;
        a.push_back(x.mean());
        x = x * z;
        a.push_back(x.mean());
        b = y.moments(4);
    }
    perf.finish_line(MC_TEST_REPEATS,
                     "test_58_random_pack_moments_allinf");
    for(int k=0;k<=4;++k)
    {
        MC_INFO << fmt(MC_TEST_FIELD_STRING) % "moments allinf: ";
        numerical_report(k, a[k], b[k]);
        BOOST_CHECK(abs(a[k] - b[k])<= MC_TEST_EPSILON);
        // BOOST_CHECK(a[k] == b[k]);
    }
}
