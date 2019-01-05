//
// Created by asem on 09/09/18.
//

#ifndef MARKOVIAN_FEATURES_HISTOGRAM_HPP
#define MARKOVIAN_FEATURES_HISTOGRAM_HPP

#include "common.hpp"

namespace buffers
{
template < size_t Size = 0 >
class Histogram
{
    static constexpr double eps = std::numeric_limits<double>::epsilon();
public:
    static constexpr double PseudoCounts = double( 0.1 ) / (Size + eps);

    using Buffer = std::vector<double>;
    using BufferIterator = typename Buffer::iterator;
    using BufferConstIterator = typename Buffer::const_iterator;

public:
    template < size_t N = Size ,
            typename std::enable_if<N != 0 , int>::type = 0 >
    explicit Histogram( double pseudoCount = PseudoCounts )
    {
        static_assert( N > 0 , "N must not be zero!" );
        _buffer = std::vector<double>( N , pseudoCount );
    }

    template < size_t N = Size ,
            typename std::enable_if<N == 0 , int>::type = 0 >
    explicit Histogram( size_t size , double pseudoCount = PseudoCounts )
    {
        assert( size > 0 );
        _buffer = std::vector<double>( size , pseudoCount );
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
        ++_buffer.at( state );
    }

    inline void increment( size_t state , double val )
    {
        _buffer.at( state ) += val;
    }

    inline double sum() const
    {
        return std::accumulate( _buffer.cbegin() , _buffer.cend() ,
                                double( 0 ));
    }

    inline void normalize()
    {
        const auto total = sum();
        for ( auto &p : _buffer )
            p /= total;
    }

    constexpr static auto maxInformation( size_t size )
    {
        return maxInformation_UNIFORM( size );
    }

    inline double information() const
    {
        return entropy( _buffer.cbegin() , _buffer.cend());
    }

    std::string toString() const
    {
        return io::join2string( _buffer , " " );
    }

    inline size_t size() const
    {
        return _buffer.size();
    }

    Histogram &operator+=( const Histogram &other )
    {
        assert( size() == other.size());
        for ( auto i = 0; i < _buffer.size(); ++i )
            _buffer[i] += (other._buffer[i]);
        return *this;
    }

    Histogram operator*( double factor ) const
    {
        Histogram newKernel;
        for ( auto i = 0; i < size(); ++i )
            newKernel._buffer[i] = _buffer[i] * factor;
        return newKernel;
    }

    Histogram operator-( const Histogram &other ) const
    {
        assert( size() == other.size());
        Histogram diff;
        for ( auto i = 0; i < _buffer.size(); ++i )
            diff._buffer[i] = _buffer[i] - other._buffer[i];
        return diff;
    }

    static inline double dot( const Histogram &k1 , const Histogram &k2 )
    {
        assert( k2.size() == k1.size());
        double sum = 0;
        for ( auto i = 0; i < k1.size(); ++i )
            sum += k1._buffer[i] * k2._buffer[i];
        return sum;
    }

    static inline double magnitude( const Histogram &k )
    {
        double sum = 0;
        for ( auto i = 0; i < k.size(); ++i )
            sum += k._buffer[i] * k._buffer[i];
        return std::sqrt( sum );
    }

    inline double magnitude() const
    {
        return magnitude( *this );
    }

    double operator*( const Histogram &other ) const
    {
        return dot( *this , other );
    }

protected:
    Buffer _buffer;
};

template < size_t Size = 0 >
class BooleanHistogram
{
public:
    using Buffer = std::vector<bool>;
    using BufferIterator = typename Buffer::iterator;
    using BufferConstIterator = typename Buffer::const_iterator;

public:
    template < size_t N = Size ,
            typename std::enable_if<N != 0 , int>::type = 0 >
    BooleanHistogram()
            : _buffer( N , false )
    {
        static_assert( N > 0 , "N must not be zero!" );
    }

    explicit BooleanHistogram( size_t size )
            : _buffer( size , false )
    {
        assert( size > 0 );
    }

    inline size_t size() const
    {
        return _buffer.size();
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

    inline auto at( size_t state ) const
    {
        return _buffer.at( state );
    }

    inline bool operator[]( size_t state ) const
    {
        return _buffer[state];
    }

    inline void set( size_t state )
    {
        _buffer.at( state ) = true;
    }

    inline double sum() const
    {
        return std::accumulate( _buffer.cbegin() , _buffer.cend() ,
                                double( 0 ));
    }

    template < typename HistogramT >
    static Histogram<Size> accumulate( HistogramT &&histogram , const BooleanHistogram &bHistogram )
    {
        auto hist = std::forward<HistogramT>( histogram );
        assert( hist.size() == bHistogram.size());
        std::transform( hist.begin() , hist.end() , bHistogram.begin() , hist.begin() , std::plus<>());
        return hist;
    }

    static Histogram<Size> accumulate( const std::vector<BooleanHistogram> &bHistograms )
    {
        return std::accumulate( bHistograms.cbegin() , bHistograms.cend() ,
                                Histogram<Size>( bHistograms.front().size()) ,
                                []( Histogram<Size> &&hist , const BooleanHistogram<Size> &bHist )
                                {
                                    return accumulate( std::move( hist ) , bHist );
                                } );
    }

protected:
    Buffer _buffer;
};

}

#endif //MARKOVIAN_FEATURES_HISTOGRAM_HPP
