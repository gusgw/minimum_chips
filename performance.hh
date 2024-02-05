/*****************************************************************************
 * Measure performance with the system clock                                 *
 * -----------------------------------------                                 *
 *                                                                           *
 * Use std::chrono to time execution, if necessary recording MPI rank as     *
 * well as time elapsed.                                                     *
 ****************************************************************************/

// Guards to prevent repeated inclusion of this file.
#ifndef MC_PERFORMANCE
#define MC_PERFORMANCE

#include <fstream>
#include <chrono>
#include <boost/format.hpp>

using perf_clock = std::chrono::steady_clock;
using perf_second = std::chrono::duration<double,std::ratio<1>>;

struct timer {
	std::string output_file_name;
	std::chrono::time_point<perf_clock> start;
	std::ofstream perf_stream;
	int np = 1;
	int rank = 0;
	std::string nth = "1";

	timer(std::string f) : output_file_name{f}
	{
#ifdef MC_OMP
		nth = std::string(getenv("OMP_NUM_THREADS"));
#endif
		perf_stream.open(output_file_name, std::ios_base::app);
	}

	~timer()
	{
		perf_stream.close();
	}

	void start_timer()
	{
		start = perf_clock::now();
	}

	void finish_line()
	{
		finish_line(1, "unknown");
	}

	void finish_line(int repeats)
	{
		finish_line(repeats, "unknown");
	}

	void finish_line(std::string tag)
	{
		finish_line(1, tag);
	}

	void finish_line(int repeats, std::string tag)
	{
		perf_second elapsed {perf_clock::now() - start};
		double seconds_elapsed {elapsed.count()};
		double seconds_per_repeat {seconds_elapsed / repeats};

		perf_stream << boost::format("%s") % nth << "\t";
		perf_stream << boost::format("%d") % rank << "\t";
		perf_stream << boost::format("%d") % np << "\t";
		perf_stream << boost::format("%d") % repeats << "\t";
		perf_stream << boost::format("%16.8e") % seconds_elapsed << "\t";
		perf_stream << boost::format("%16.8e") % seconds_per_repeat << "\t";
		perf_stream << tag << std::endl;
	}
};

#endif
