//
// Created by asem on 09/09/18.
//

#ifndef MARKOVIAN_FEATURES_HISTOGRAM_HPP
#define MARKOVIAN_FEATURES_HISTOGRAM_HPP

#include "common.hpp"

namespace buffers {
    template<size_t StatesN>
    class Histogram
    {
    public:
        static constexpr double PseudoCounts = double( 1 ) / StatesN;

        using Buffer = std::array<double, StatesN>;
        using BufferIterator = typename Buffer::iterator;
        using BufferConstIterator = typename Buffer::const_iterator;

    public:
        explicit Histogram( double pseudoCount = PseudoCounts )
                : _hits( 0 )
        {
            _buffer.fill( pseudoCount );
        }

        inline BufferConstIterator begin() const
        {
            return _buffer.begin();
        }

        inline BufferConstIterator end() const
        {
            return _buffer.end();
        }

        inline BufferConstIterator cbegin() const
        {
            return _buffer.cbegin();
        }

        inline BufferConstIterator cend() const
        {
            return _buffer.cend();
        }

        inline auto &at( size_t state ) const
        {
            return _buffer.at( state );
        }

        inline double operator[]( size_t state ) const
        {
            return _buffer[state];
        }

        inline void increment( size_t state )
        {
            ++_hits;
            ++_buffer.at( state );
        }

        inline void increment( size_t state, double val )
        {
            ++_hits;
            _buffer.at( state ) += val;
        }

        inline double sum() const
        {
            return std::accumulate( _buffer.cbegin(), _buffer.cend(),
                                    double( 0 ));
        }

        inline void normalize()
        {
            const auto total = sum();
            for (auto &p : _buffer)
                p /= total;
        }

        constexpr static auto maxInformation()
        {
            return maxInformation_UNIFORM( StatesN );
        }

        inline double information() const
        {
            return entropy( _buffer.cbegin(), _buffer.cend());
        }

        inline size_t hits() const
        {
            return _hits;
        }

        std::string toString() const
        {
            return io::join2string( _buffer, " " );
        }

        Histogram &operator+=( const Histogram &other )
        {
            for (auto i = 0; i < StatesN; ++i)
                _buffer[i] += (other._buffer[i]);
            return *this;
        }

        Histogram operator*( double factor ) const
        {
            Histogram newKernel;
            for (auto i = 0; i < StatesN; ++i)
                newKernel._buffer[i] = _buffer[i] * factor;
            return newKernel;
        }

        Histogram operator-( const Histogram &other ) const
        {
            Histogram diff;
            for( auto i = 0; i < StatesN; ++i )
                diff._buffer[i] = _buffer[i] - other._buffer[i];
            return diff;
        }

        static inline double dot( const Histogram &k1, const Histogram &k2 )
        {
            double sum = 0;
            for (auto i = 0; i < StatesN; ++i)
                sum += k1._buffer[i] * k2._buffer[i];
            return sum;
        }

        static inline double magnitude( const Histogram &k )
        {
            double sum = 0;
            for (auto i = 0; i < StatesN; ++i)
                sum += k._buffer[i] * k._buffer[i];
            return std::sqrt( sum );
        }

        inline double magnitude() const
        {
            return magnitude( *this );
        }

        double operator*( const Histogram &other ) const
        {
            return dot( *this, other );
        }

    protected:
        Buffer _buffer;
        size_t _hits;
    };

}

#endif //MARKOVIAN_FEATURES_HISTOGRAM_HPP
