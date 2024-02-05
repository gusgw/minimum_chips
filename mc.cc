#include <iostream>
#include <stdexcept>

// BOOST
#include <boost/format.hpp>
using fmt = boost::format;

// Turn on message passing
// define MC_MPI

// Switch on some debugging code that keeps
// track of the generation of random numbers.
#define MC_ACCUMULATE_RANDOM
#define MC_SAVE_RANDOM

// Switch on fixed seed so we get the same pseudorandom
// sequence every time
#define MC_REPRODUCE

// Use 10 repeats if not defined in the Makefile
#ifndef MC_REPEATS
#define MC_REPEATS 10
#endif

#include "log.hh"       // Set up simple logging
#include "machine.hh"   // Set up and wrap pseudorandom numbers

// Split an array between MPI processes,
// and implement arithmetic
#include "pack.hh"

// Measure performance
#include "performance.hh"

int main(int argc, char **argv)
{
    // Any setup for logging
    start_logging();

    // Store array sizes and distribution
    // across processes
    dimensions *dim;
    std::vector<size_t> NN = {1};
    dim = new dimensions(argc, argv, NN);

    // Start performance timer
    timer perf("timing.tsv");

    // Store test statistics from random array
    double mn, mn2;

    // Set up the pseudorandom number generator
    machine<float,
            std::mt19937_64,
            std::normal_distribution<float>> nrm(0.0, 1.0);

    // Construct a random array
    random_pack<float,
                double,
                float,
                std::mt19937_64,
                std::normal_distribution<float>> x(dim->sz, nrm);

    // Start timing for numerical work
    perf.start_timer();

    // Distributed averages
    for(int k=0; k<MC_REPEATS; ++k)
    {
        mn = x.mean();
        mn2 = x.mean_square();
    }

    // Stop the performance timer
    perf.finish_line(MC_REPEATS, "statistics_from_mc");

    // Report statistics from 0th process when all done
    {
        MC_INFO << fmt("%16s") % "mean: " << fmt("%15.8e") % mn;
        MC_INFO << std::endl;
        MC_INFO << fmt("%16s") % "mean square: " << fmt("%15.8e") % mn2;
        MC_INFO << std::endl;
    }

    // Discard array sizes and distribution across processes
    delete dim;

    return 0;
}
