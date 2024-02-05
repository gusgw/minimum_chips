/**
 * @file log.hh
 *
 * @brief Set up logging
 *
 * Use trivial logging. Implementation of different logging levels is not
 * yet done, but it should be possible to keep macros like MC_INFO when that
 * is done.
 */

// Guards to prevent repeated inclusion of this file
#ifndef MC_LOG
#define MC_LOG

#include <iostream>
#include <boost/format.hpp>
using fmt = boost::format;

std::ostream null(nullptr);

class tagged_stream {
public:
    tagged_stream(std::string tt, int pp, int bb, int qq)
        : tag{tt}, p{pp}, b{bb}, q{qq}, new_line(true), on(false) {}

    void label(int pp, int bb, int  qq)
    {
        p = pp;
        b = bb;
        q = qq;
    }

    template<class T>
    tagged_stream &operator<<(T val)
    {
        if(new_line)
        {
            std::clog << tag;
            if(p==-1)
                std::clog << "p-";
            else
                std::clog << fmt("p%03d") % p;
            if(b==-1)
                std::clog << "b-";
            else
                std::clog << fmt("b%03d") % b;
            if(q==-1)
                std::clog << "q-: ";
            else
                std::clog << fmt("q%03d ") % q;
            new_line = false;
        }
        std::clog << val;
        return *this;
    }

    tagged_stream &operator<<(std::ostream& (*fn)(std::ostream&))
    {
        std::clog << fn;
        if(fn == (std::ostream& (*)(std::ostream&)) std::endl)
        {
            new_line = true;
        }
        return *this;
    }
private:
    std::string tag;
    int p;
    int b;
    int q;
    bool new_line;
public:
    bool on;
};

tagged_stream debug_stream("dbug:", -1, -1, -1);
tagged_stream info_stream("info:", -1, -1, -1);
tagged_stream warn_stream("warn:", -1, -1, -1);

#define MC_DEBUG null
#define MC_INFO info_stream
#define MC_WARN warn_stream

void start_logging()
{
    MC_INFO <<  "log started" << std::endl;
}

#endif // MC_LOG