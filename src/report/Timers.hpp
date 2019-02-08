//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_TIMERS_HPP
#define MARKOVIAN_FEATURES_TIMERS_HPP

#include "common.hpp"

class Timers
{
private:
    using Time = decltype( std::chrono::system_clock::now());
    using Diff = std::chrono::milliseconds;
    struct Clock
    {
        Time c1;
        Time c2;
        Diff diff{Diff::zero()};
    };

    using TimersDictionary = std::map<std::string, Clock>;

    static TimersDictionary &_timers()
    {
        static TimersDictionary singleton;
        return singleton;
    }

    static const TimersDictionary &_ctimers()
    {
        return _timers();
    }

    static Time &_timer1( const char *label )
    {
        return _timers()[label].c1;
    }

    static Time &_timer2( const char *label )
    {
        return _timers()[label].c2;
    }

    static Diff &_accumulator( const char *label )
    {
        return _timers()[label].diff;
    }

public:
    static void tic( const char *label )
    {
        fmt::print( "[{}...]\n", label );
        _timer1( label ) = std::chrono::system_clock::now();
    }

    static void toc( const char *label )
    {
        fmt::print( "[DONE][{}]\n", label );
        _timer2( label ) = std::chrono::system_clock::now();
        _accumulator( label ) +=
                std::chrono::duration_cast<std::chrono::milliseconds>( _timer2( label ) - _timer1( label ));
    }

    static auto duration_s( const char *label )
    {
        auto diff =
                std::chrono::duration_cast<std::chrono::seconds>(
                        _accumulator( label ));
        return diff.count();
    }

    static auto duration_ms( const char *label )
    {
        auto diff =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        _accumulator( label ));
        return diff.count();
    }

    static void report_s( const char *label )
    {
        fmt::print( "[Time elapsed for {}:{} seconds]\n", label, duration_s( label ));
    }

    static void report_ms( const char *label )
    {
        fmt::print( "[Time elapsed for {}:{} msec]\n", label, duration_ms( label ));
    }

    template<typename ReportedFunction>
    static auto reported_invoke_s(
            ReportedFunction fn,
            const char *label
    )
    {
        tic( label );
        auto ret = fn();
        toc( label );
        report_s( label );
        return ret;
    }
};


#endif //MARKOVIAN_FEATURES_TIMERS_HPP
