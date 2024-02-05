/**
 * @file machine.hh
 *
 * @brief Random number generator template
 * 
 * A template switches between types, generators, and distributions.
 * The switch MC_REPRODUCE forces the reproduction of the same pseudorandom
 * sequence. The MC_ACCUMULATE_RANDOM switch keeps a sum of generated random
 * numbers, and a sum of their squares. The MC_SAVE_RANDOM switch causes a
 * complete history of the random numbers generated to be saved.
 *
 * The template below is intended to work with the following distributions:
 * -   Normal
 * -   Log normal
 * -   \Gamma
 * -   Uniform real distribution
 * -   Exponential distribution
 *
 * Note that the variables Sum and SumSquare, used to accumulate statistics
 * are long double, intended to accumulate random numbers usually of type
 * float or double. It may be possible to use the template to generate
 * random integers.
 */

// Guards to prevent repeated inclusion of this file
// and its template methods in the corresponding .tt file.
#ifndef MC_MACHINE
#define MC_MACHINE

// The seed is set to different values
// for different MPI processes. A call to
// MPI_Comm_rank(...) is used.

// Seed to use when generating the same pseudorandom
// number sequence repeatedly for debugging purposes.
#define FIXED_SEED 0

// See https://cplusplus.com/reference/random/
#include <random>

//       number: Usually this should be a floating point type.
//       engine: Choose a random number generator,
//               for example std::mt19937_64, obtained from <random>.
// distribution: Choose a distribution, for example
//               std::normal_distribution<double>, as discussed above.
template<typename number, typename engine, typename distribution>
class machine {
public:
    // Constructors that accept parameters with the same type as
    // the generated random samples:
    machine(number p);           // Construct for one-parameter distributions
    machine(number p, number q); // Construct for two-parameter distributions
    ~machine();                  // Destructor
    void SetSeed();              // Set seed from clock or to a fixed value
    number operator()();         // Overload () so it samples a random number
protected:
    unsigned seed;               // Store the random seed.
    engine* generator = nullptr; // Point to the random number generator
    distribution* rho = nullptr; // Point to the distribution
#ifdef MC_ACCUMULATE_RANDOM
    unsigned Count = 0;          // Count random numbers returned
    long double Sum = 0.0;       // Sum of random numbers so far
    long double SumSquare = 0.0; // Sum of squares of random numbers
#endif
#ifdef MC_SAVE_RANDOM
    std::vector<number> history; // Save all random numbers generated
#endif
};

/*****************************************************************************/

// Template methods must be defined in the header,
// so include them here to keep the .hh file short.
#include "machine.tt"

#endif
