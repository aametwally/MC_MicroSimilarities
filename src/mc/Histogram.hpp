//
// Created by asem on 09/09/18.
//

#ifndef MARKOVIAN_FEATURES_HISTOGRAM_HPP
#define MARKOVIAN_FEATURES_HISTOGRAM_HPP

#include "common.hpp"

namespace buffers {
template<size_t Size = 0>
class Histogram
{
    static constexpr double eps = std::numeric_limits<double>::epsilon();
public:
    using Buffer = std::vector<double>;
    using BufferIterator = typename Buffer::iterator;
    using BufferConstIterator = typename Buffer::const_iterator;
    static constexpr size_t HistogramSize = Size;

    using ValueType = double;
public:
    template<size_t N = Size,
            typename std::enable_if<N != 0, int>::type = 0>
    explicit Histogram( double pseudoCount )
    {
        static_assert( N > 0, "N must not be zero!" );
        _buffer = std::vector<double>( N, pseudoCount );
    }

    template<size_t N = Size,
            typename std::enable_if<N == 0, int>::type = 0>
    explicit Histogram(
            size_t size,
            double pseudoCount
    )
    {
        static_assert( N == 0, "Size paramter must be zero." );
        assert( size > 0 );
        _buffer = std::vector<double>( size, pseudoCount );
    }

    static Histogram ones()
    {
        Histogram<Size> h( 1.0 );
        return h;
    }

    static Histogram zeros()
    {
        Histogram<Size> h( 0.0 );
        return h;
    }

    inline BufferIterator begin()
    {
        return _buffer.begin();
    }

    inline BufferIterator end()
    {
        return _buffer.end();
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

    inline void increment(
            size_t state,
            double val
    )
    {
        _buffer.at( state ) += val;
    }

    inline double sum() const
    {
        return std::accumulate( _buffer.cbegin(), _buffer.cend(),
                                double( 0 ));
    }

    inline Histogram &normalize()
    {
        const auto total = sum();
        for (auto &p : _buffer)
            p /= total;
        return *this;
    }

    inline Histogram &unitVectorNormalize()
    {
        auto radius = magnitude();
        return this->operator/=( radius );
    }

    constexpr static auto maxInformation( size_t size )
    {
        return maxInformation_UNIFORM( size );
    }

    inline double information() const
    {
        return entropy( _buffer.cbegin(), _buffer.cend());
    }

    std::string toString() const
    {
        return io::join2string( _buffer, " " );
    }

    inline size_t size() const
    {
        return _buffer.size();
    }

    Histogram &operator+=( const Histogram &other )
    {
        assert( size() == other.size());
        for (auto i = 0; i < _buffer.size(); ++i)
            _buffer[i] += (other._buffer[i]);
        return *this;
    }

    Histogram &operator-=( const Histogram &other )
    {
        assert( size() == other.size());
        for (auto i = 0; i < _buffer.size(); ++i)
            _buffer[i] -= (other._buffer[i]);
        return *this;
    }

    Histogram &operator*=( double scale )
    {
        for (auto i = 0; i < _buffer.size(); ++i)
            _buffer[i] *= scale;
        return *this;
    }

    Histogram &operator/=( double scale )
    {
        for (auto i = 0; i < _buffer.size(); ++i)
            _buffer[i] /= scale;
        return *this;
    }

    Histogram &square()
    {
        for (auto &val : _buffer)
            val *= val;
        return *this;
    }

    Histogram &sqrt()
    {
        for (auto &val : _buffer)
            val = std::sqrt( val );
        return *this;
    }

    Histogram operator*( double factor ) const
    {
        auto newKernel = zeros();
        for (auto i = 0; i < size(); ++i)
            newKernel._buffer[i] = _buffer[i] * factor;
        return newKernel;
    }

    Histogram operator-( const Histogram &other ) const
    {
        assert( size() == other.size());
        Histogram diff( 0 );
        assert( diff.size() == size());
        for (auto i = 0; i < _buffer.size(); ++i)
            diff._buffer[i] = _buffer[i] - other._buffer[i];
        return diff;
    }

    static inline double dot(
            const Histogram &k1,
            const Histogram &k2
    )
    {
        assert( k2.size() == k1.size());
        double sum = 0;
        for (auto i = 0; i < k1.size(); ++i)
            sum += k1._buffer[i] * k2._buffer[i];
        return sum;
    }

    static inline double magnitude( const Histogram &k )
    {
        double sum = 0;
        for (auto i = 0; i < k.size(); ++i)
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

    void swap( Histogram &other )
    {
        _buffer.swap( other._buffer );
    }

    void clear()
    {
        _buffer.clear();
    }

    static Histogram<Size> mean(
            const std::vector<Histogram<Size>> &histograms,
            size_t n
    )
    {
        static_assert( Size > 0, "Histogram size must be non-zero" );

        assert( !histograms.empty());

        auto accumulative = std::accumulate( histograms.cbegin(), histograms.cend(),
                                             Histogram<Size>( 0.0 ),
                                             [](
                                                     Histogram<Size> hist,
                                                     const Histogram<Size> &bHist
                                             ) {
                                                 return std::move( hist += bHist );
                                             } );
        return std::move( accumulative /= double( n ));
    }


    static Histogram<Size> variance(
            const std::vector<Histogram<Size>> &histograms,
            const Histogram<Size> &mean,
            size_t n
    )
    {
        static_assert( Size > 0, "Histogram size must be non-zero" );
        using T = typename Histogram<Size>::ValueType;
        assert( n > 0 );

        Histogram<Size> variance( 0.0 );
        auto sumSquares = std::accumulate(
                histograms.cbegin(), histograms.cend(),
                Histogram<Size>( 0.0 ),
                [&](
                        Histogram<Size> hist,
                        const Histogram<Size> &bHist
                ) {
                    hist += (bHist - mean).square();
                    return std::move( hist );
                } );
        return std::move( sumSquares /= double( n ));
    }

    static Histogram<Size> variance(
            const std::vector<Histogram<Size>> &histograms,
            size_t n
    )
    {
        return variance( histograms, mean( histograms, n ), n );
    }

    static Histogram<Size> standardDeviation(
            const std::vector<Histogram<Size>> &histograms,
            size_t n
    )
    {
        return variance( histograms, n ).sqrt();
    }

    static Histogram<Size> standardDeviation(
            const std::vector<Histogram<Size>> &histograms,
            const Histogram<Size> &meanHistogram,
            size_t n
    )
    {
        return variance( histograms, meanHistogram, n ).sqrt();
    }

    static Histogram<Size> standardError(
            const std::vector<Histogram<Size>> &histograms,
            const Histogram<Size> &meanHistogram,
            size_t n
    )
    {
        return variance( histograms, meanHistogram, n ).sqrt() * (1 + 1 / std::sqrt( histograms.size()));
    }

protected:
    Buffer _buffer;
};

template<size_t Size = 0>
class BooleanHistogram
{
public:
    using Buffer = std::vector<bool>;
    using BufferIterator = typename Buffer::iterator;
    using BufferConstIterator = typename Buffer::const_iterator;

public:
    template<size_t N = Size,
            typename std::enable_if<N != 0, int>::type = 0>
    BooleanHistogram()
            : _buffer( N, false )
    {
        static_assert( N > 0, "N must not be zero!" );
    }

    explicit BooleanHistogram( size_t size )
            : _buffer( size, false )
    {
        assert( size > 0 );
    }

    explicit BooleanHistogram( Buffer data )
            : _buffer( std::move( data ))
    {
        assert( _buffer.size() > 0 );
    }

    template<typename BooleanHistogramType>
    static inline Histogram<Size> toDoubleHistogram( BooleanHistogramType &&hist )
    {
        auto dHist = Histogram<Size>::zeros();
        std::transform( std::cbegin( hist ), std::cend( hist ), dHist.cbegin(), dHist.begin(),
                        [](
                                bool a,
                                double
                        ) {
                            return static_cast<double>(a);
                        } );
        return dHist;
    }

    inline Histogram<Size> unitVectorNormalized() const
    {
        auto dHist = toDoubleHistogram( *this );
        return std::move( dHist.unitVectorNormalize());
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
        return std::accumulate( _buffer.cbegin(), _buffer.cend(),
                                double( 0 ));
    }

    template<typename HistogramT>
    static Histogram<Size> accumulate(
            HistogramT &&histogram,
            const BooleanHistogram<Size> &bHistogram
    )
    {
        using T = typename HistogramT::ValueType;
        Histogram<Size> hist = std::forward<HistogramT>( histogram );
        assert( hist.size() == bHistogram.size());
        std::transform( hist.begin(), hist.end(), bHistogram.cbegin(), hist.begin(),
                        [](
                                T a,
                                bool b
                        ) -> T {
                            return a + b;
                        } );
        return hist;
    }

    static Histogram<Size> accumulate( const std::vector<BooleanHistogram<Size>> &bHistograms )
    {
        static_assert( Size > 0, "Histogram size must be non-zero" );
        return std::accumulate( bHistograms.cbegin(), bHistograms.cend(),
                                Histogram<Size>( double( 0 )),
                                [](
                                        Histogram<Size> hist,
                                        const BooleanHistogram<Size> &bHist
                                ) {
                                    return accumulate( std::move( hist ), bHist );
                                } );
    }

    static Histogram<Size> varianceBernoulli(
            const Histogram<Size> &mean,
            size_t n
    )
    {
        using T = typename Histogram<Size>::ValueType;
        auto variance = Histogram<Size>::zeros();
        assert( mean.size() == variance.size());
        std::transform( mean.cbegin(), mean.cend(), variance.cbegin(), variance.begin(),
                        [=](
                                T p,
                                T
                        ) {
                            return p * (1 - p);
                        } );
        return variance;
    }

    static Histogram<Size> standardDeviationBernoulli(
            const Histogram<Size> &mean,
            size_t n
    )
    {
        return varianceBernoulli( mean, n ).sqrt();
    }

    template<typename HistogramT>
    static BooleanHistogram<Size>
    binarizeHistogram(
            HistogramT &&histogram,
            typename HistogramT::ValueType threshold = 0
    )
    {
        auto hist = std::forward<HistogramT>( histogram );
        Buffer bBuffer;
        std::transform( std::begin( hist ), std::end( hist ),
                        std::back_inserter( bBuffer ), [=]( typename HistogramT::ValueType value ) -> bool {
                    return value > threshold;
                } );
        assert( bBuffer.size() == hist.size());
        return BooleanHistogram<Size>( bBuffer );
    }

    void swap( BooleanHistogram &other )
    {
        _buffer.swap( other._buffer );
    }

    void clear()
    {
        _buffer.clear();
    }

protected:
    Buffer _buffer;
};

}

#endif //MARKOVIAN_FEATURES_HISTOGRAM_HPP
