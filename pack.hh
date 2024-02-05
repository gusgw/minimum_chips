/*****************************************************************************
 * Set up arrays that describe random fields                                 *
 * -----------------------------------------                                 *
 * Store a tensor field, or similar structure, in an array and provide       *
 * efficient operations on the field. Allow for splitting the field across   *
 * MPI processes.                                                            *
 *****************************************************************************/

// Guards to prevent repeated inclusion of this file
// and its template methods in the corresponding .tt file.
#ifndef MC_FIELDS
#define MC_FIELDS

#include <boost/assert.hpp>
#include <boost/format.hpp>

#include <cmath>
#include <stdexcept>
#include <omp.h>

// Optionally split the fields between MPI processes
// mpi.h is probably included with main().

#include "log.hh"
#include "machine.hh"

using std::size_t;
using fmt = boost::format;

/*****************************************************************************/

// Store the number of components in each dimension.
class dimensions {
public:
    size_t sz;                              // Vector components per process
    size_t M;                               // Vector components

    int np = 1;                             // Number of processes
    int nb = 1;                             // Number of blocks
    int nq = 1;                             // Processes per block

    int prank = 0;                          // Overall process rank
    int block = 0;                          // Block number
    int qrank = 0;                          // Rank within block

    std::vector<size_t> N;                  // Store a vector of sizes


    // Construct the array sizes and distribution of data and processes
    dimensions(int argc,
               char **argv,
               std::vector<size_t>& NN) : N{NN}
    {
        command_line(argc, argv);
        start(argc, argv);
        set_rank();
        set_block();
    }

    // Construct the array sizes and distribution of data and processes
    dimensions(size_t MM,
               std::vector<size_t>& NN) : M{MM}, N{NN}
    {
        nq = 1;
        start();
        set_rank();
        set_block();
    }

    void command_line(int argc, char **argv)
    {
        if(argc<2)
            throw std::invalid_argument("missing command line argument");
        else
        {
            M = atoi(argv[1]);
            MC_INFO << fmt("%16s") % "M: " << fmt("%d") % M << std::endl;
        }
    }

    void start()
    {

        MC_INFO << "nothing to do for start() if MPI is off" << std::endl;

    }

    void start(int argc, char **argv)
    {

        MC_INFO << "nothing to do for start()" << std::endl;

    }

    void set_rank()
    {

        MC_INFO << "nothing to do for rank()" << std::endl;

    }

    void set_block()
    {
        sz = M;
        MC_INFO << "only setting sz in set_block()" << std::endl;
        MC_DEBUG << fmt("%16s") % "sz: " << fmt("%d") % sz << std::endl; 
    }

    ~dimensions()
    {

        MC_INFO << "nothing to do for ~dimensions()" << std::endl;

    }
};

/*****************************************************************************/

// All three template arguments are expected to be floating
// point types. They may all be the same.
//    scalar: Use this type for usual scalars, inner products etc. 
//       sum: Use a higher precision type for sums and similar.
// component: Use this type for the components of the array.
// Most of this base type is as described by Stroustrup in
// 'A Tour of C++', 2nd ed., Addison-Wesley, Boston 2018.
template<typename scalar, typename sum, typename component>
class pack {
public:
    pack(size_t s);                         // Constructor
    pack(const pack& v);                    // Copy constructor
    pack(pack&& v);                         // Move constructor
    ~pack();                                // Destructor
    pack& operator=(const pack& v);         // Copy assignment
    pack& operator=(const scalar& a);
    pack& operator=(pack&& v);              // Move assignment
    pack& operator=(scalar&& a);
    // Assignment operators with arithmetic
    pack& operator+=(const pack& v);
    pack& operator+=(pack&& v);
    pack& operator+=(const scalar& a);
    pack& operator+=(scalar&& a);
    pack& operator-=(const pack& v);
    pack& operator-=(pack&& v);
    pack& operator-=(const scalar& a);
    pack& operator-=(scalar&& a);
    pack& operator*=(const pack& v);
    pack& operator*=(pack&& v);
    pack& operator*=(const scalar& a);
    pack& operator*=(scalar&& a);
    pack& operator/=(const pack& v);
    pack& operator/=(pack&& v);
    pack& operator/=(const scalar& a);
    pack& operator/=(scalar&& a);
    // Unary operators use elementwise arithmetic
    pack operator-() const;
    pack operator++();
    pack operator--();
    // Access the array
    component& operator[](size_t j); // TODO Ordinary access
    component& operator()(size_t j); // TODO Use full array index
    // Essential mathematical operations
    void abs();
    void square();
    void reciprocal();
    void power(int m);
    void exp();
    void mayerf(scalar beta);
    void apply(component (*f)(component));
    // Statistical properties
    sum mean();
    sum mean_square();
    sum rms();
    sum mean_abs();
    sum mean_apply(component (*f)(component));

    sum local_mean(bool all);
    sum local_mean();
    sum local_mean_square(bool all);
    sum local_mean_square();
    sum local_rms(bool all);
    sum local_rms();
    sum local_mean_abs(bool all);
    sum local_mean_abs();
    sum local_mean_apply(bool all, component (*f)(component));
    sum local_mean_apply(component (*f)(component));

    std::vector<sum> moments(int m);
    // Array properties
    size_t sz = 0;                      // Store number of components
    size_t FullSize = 0;
    size_t Start = 0;
    component* x = nullptr;             // Components of the tensor
    component* localx = nullptr;        // Pointer for MPI_Win_allocate_shared,
                                        // same as x for serial calculations.
    // Elementwise arithmetic is the default
    // The % operator is defined as the same elementwise product
    // as the * operator, as it may later serve for convolution.
    pack operator+(const pack& v) const;
    pack operator-(const pack& v) const;
    pack operator*(const pack& v) const;
    pack operator%(const pack& v) const;
    pack operator/(const pack& v) const;
    // Scalar operations
    // Corresponding operations with the scalar given
    // first are overloaded by functions defined in Field.tt.
    pack operator+(const scalar& a) const;
    pack operator-(const scalar& a) const;
    pack operator*(const scalar& a) const;
    pack operator/(const scalar& a) const;
};

/*****************************************************************************/

template<typename scalar,
         typename sum,
         typename component,
         typename engine,
         typename distribution>
class random_pack : public pack<scalar, sum, component> {
public:
    random_pack(size_t s, machine<component,
                                  engine,
                                  distribution>& random) : pack<scalar,
                                                                sum,
                                                                component>(s)
    {
        for(size_t i=0; i<this->sz; ++i)
            (this->x)[i] = random();
    }
    random_pack(pack<scalar, sum, component>&& v)
        : pack<scalar, sum, component>::pack(v) {}
    random_pack& operator=(pack<scalar, sum, component>&& v);
    random_pack(scalar&& a)
        : pack<scalar, sum, component>::pack(a) {}
    random_pack& operator=(scalar&& a);
};

/*****************************************************************************/

#include "pack.tt"

#endif

/*****************************************************************************/
