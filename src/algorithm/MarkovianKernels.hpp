#ifndef MARKOVIAN_KERNELS_HPP
#define MARKOVIAN_KERNELS_HPP


#include "common.hpp"

#include "Series.hpp"
#include "similarities.hpp"
#include "aminoacids_grouping.hpp"

auto geometricDistribution( double p )
{
    return [p]( double exponent ) {
        return pow( p, exponent );
    };
}

template<typename T>
auto inverseFunction()
{
    return []( T n ) {
        return T( 1 ) / n;
    };
}

template<typename AAGrouping>
class MarkovianKernels
{
public:
    using Order = int8_t;
    using KernelID = size_t;

    static constexpr size_t StatesN = AAGrouping::StatesN;
    static constexpr std::array<char, StatesN> ReducedAlphabet = reducedAlphabet<StatesN>();
    static constexpr std::array<char, 256> ReducedAlphabetIds = reducedAlphabetIds( AAGrouping::Grouping );
    static constexpr Order MinOrder = 3;
    static constexpr double PseudoCounts = double( 1 ) / StatesN;

    using Buffer = std::array<double, StatesN>;
    using BufferIterator = typename Buffer::iterator;
    using BufferConstIterator = typename Buffer::const_iterator;

public:
    class Kernel
    {
    public:
        explicit Kernel( double pseudoCount = PseudoCounts )
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

        inline auto &at( char state ) const
        {
            return _buffer.at( state );
        }

        inline void increment( char state )
        {
            ++_hits;
            ++_buffer.at( state );
        }

        inline void increment( char state, double val )
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

        Kernel &operator+=( const Kernel &other )
        {
            for (auto i = 0; i < StatesN; ++i)
                _buffer[i] += (other._buffer[i]);
            return *this;
        }

        Kernel operator*( double factor ) const
        {
            Kernel newKernel;
            for (auto i = 0; i < StatesN; ++i)
                newKernel._buffer[i] = _buffer[i] * factor;
            return newKernel;
        }

        static inline double dot( const Kernel &k1, const Kernel &k2 )
        {
            double sum = 0;
            for (auto i = 0; i < StatesN; ++i)
                sum += k1._buffer[i] * k2._buffer[i];
            return sum;
        }

        static inline double magnitude( const Kernel &k )
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

        double operator*( const Kernel &other ) const
        {
            return dot( *this, other );
        }

    protected:
        Buffer _buffer;
        size_t _hits;
    };


    template<typename T>
    using IsoKernelsAssociativeCollection = std::unordered_map<KernelID, T>;

    template<typename T>
    using HeteroKernelsAssociativeCollection = std::unordered_map<Order, IsoKernelsAssociativeCollection<T >>;

    using IsoKernels = IsoKernelsAssociativeCollection<Kernel>;
    using HeteroKernels = HeteroKernelsAssociativeCollection<Kernel>;
    using HeteroKernelsFeatures = HeteroKernelsAssociativeCollection<double>;
public:
    explicit MarkovianKernels( Order order ) :
            _maxOrder( order ),
            _hits( 0 )
    {
        assert( order >= MinOrder );
    }

    static IsoKernels
    filterPercentile( const std::unordered_map<size_t, Kernel> &filteredKernel,
                      float percentile )
    {
        std::vector<std::pair<KernelID, Kernel >> v;
        for (const auto &p : filteredKernel) v.push_back( p );

        auto cmp = []( const std::pair<size_t, Kernel> &p1, const std::pair<size_t, Kernel> &p2 ) {
            return p1.second.hits() > p2.second.hits();
        };

        size_t percentileTailIdx = filteredKernel.size() * percentile;
        std::nth_element( v.begin(), v.begin() + percentileTailIdx,
                          v.end(), cmp );

        IsoKernels filteredKernel2( v.begin(), v.begin() + percentileTailIdx );

        return filteredKernel2;
    }

    void train( const std::vector<std::string> &sequences )
    {
        for (const auto &s : sequences)
            _countInstance( s );

        for (Order order = MinOrder; order <= _maxOrder; ++order)
            for (auto &[id, kernel] : _kernels.at( order ))
                kernel.normalize();
    }

    size_t hits() const
    {
        return _hits;
    }

    void toFiles( const std::string &dir,
                  const std::string &prefix,
                  const std::string &id ) const
    {
        std::ofstream kernelFile;
        std::vector<std::string> names1 = {prefix, "profile", id};
        kernelFile.open( dir + "/" + io::join( names1, "_" ) + ".array" );
        for (const auto &[id, kernel] : kernels())
            kernelFile << kernel.toString() << std::endl;
        kernelFile.close();
    }

    const IsoKernels &kernels( Order order ) const
    {
        return _kernels.at( order );
    }

    Order maxOrder() const
    {
        return _maxOrder;
    }

    static constexpr inline KernelID lowerOrderID( KernelID id )
    { return id / StatesN; }


    template<typename Derived, typename ReturnType>
    class ObjectSeriesByOrder : public Series<ReturnType, Derived, ObjectSeriesByOrder<Derived,ReturnType>>
    {
    public:
        ObjectSeriesByOrder( const MarkovianKernels &kernels,
                             Order order,
                             KernelID id )
                : _kernels( std::cref( kernels )),
                  _mutables( order, id )
        {}

        inline bool isEmpty() const
        {
            return currentOrder() < MinOrder;
        }

        inline void popTerm()
        {
            if ( !isEmpty())
            {
                --_mutables.first;
                _mutables.second = lowerOrderID( _mutables.second );
            }
        }

        inline constexpr size_t length() const
        {
            if ( !isEmpty())
            {
                return currentOrder() + 1 - MinOrder;
            } else return 0;
        }

        inline constexpr Order currentOrder() const
        {
            return _mutables.first;
        }

        inline constexpr KernelID currentID() const
        {
            return _mutables.second;
        }

        virtual std::optional<ReturnType> currentTerm() const = 0;

    protected:
        std::reference_wrapper<const MarkovianKernels> _kernels;
        std::pair<Order, KernelID> _mutables;
    };

    class KernelSeriesByOrder : public ObjectSeriesByOrder<KernelSeriesByOrder, std::reference_wrapper<const Kernel >>
    {
    public:
        KernelSeriesByOrder( const MarkovianKernels &kernels,
                             Order order,
                             KernelID id )
                : ObjectSeriesByOrder<KernelSeriesByOrder, std::reference_wrapper<const Kernel >>( kernels, order, id )
        {}

        std::optional<std::reference_wrapper<const Kernel >> currentTerm() const override
        {
            if ( !this->isEmpty())
            {
                try
                {
                    auto order = this->_mutables.first;
                    auto id = this->_mutables.second;
                    const auto &kernel = this->_kernels.get().kernels( order ).at( id );
                    return std::cref( kernel );
                } catch (const std::out_of_range &)
                {}
            }
            return std::nullopt;
        }
    };

    class ProbabilitisByOrder : public ObjectSeriesByOrder<ProbabilitisByOrder, double>
    {
    public:
        ProbabilitisByOrder( const MarkovianKernels &kernels,
                             Order order,
                             KernelID id ) : ObjectSeriesByOrder<ProbabilitisByOrder, double>( kernels, order, id )
        {}

        std::optional<double> currentTerm() const override
        {
            if ( !this->isEmpty())
            {
                try
                {
                    auto order = this->currentOrder();
                    auto id = this->currentID();
                    const auto &kernel = this->_kernels.get().kernels( order ).at( id );
                    return double( kernel.hits()) / this->_kernels.get().hits();
                } catch (const std::out_of_range &)
                {}
            }
            return std::nullopt;
        }
    };

    inline KernelSeriesByOrder kernelsByOrder( KernelID id ) const
    {
        return KernelSeriesByOrder( *this, _maxOrder, id );
    }

    inline KernelSeriesByOrder kernelsByOrder( Order order, KernelID id ) const
    {
        return KernelSeriesByOrder( *this, order, id );
    }

    inline ProbabilitisByOrder probabilitisByOrder( Order order, KernelID id ) const
    {
        return ProbabilitisByOrder( *this, order, id );
    }

private:
    void _incrementInstance( std::string::const_iterator from,
                             std::string::const_iterator until,
                             Order order )
    {
        assert( from != until );
        KernelID id = _sequence2ID( from, until );
        auto c = ReducedAlphabetIds.at( size_t( *(until)));
        _kernels[order][id].increment( c );
    }

    void _countInstance( const std::string &sequence )
    {
        ++_hits;
        for (auto order = MinOrder; order <= _maxOrder; ++order)
            for (auto i = 0; i < sequence.size() - order; ++i)
                _incrementInstance( sequence.cbegin() + i, sequence.cbegin() + i + order, order );
    }

    static constexpr inline KernelID _char2ID( char a )
    {
        assert( a >= ReducedAlphabet.front());
        return KernelID( a - ReducedAlphabet.front());
    }

    static constexpr inline char _id2Char( KernelID id )
    {
        assert( id <= 128 );
        return char( id + ReducedAlphabet.front());
    }

    static KernelID _sequence2ID( const std::string &s )
    {
        return _sequence2ID( s.cbegin(), s.cend());
    }

    static KernelID _sequence2ID( std::string::const_iterator from,
                                  std::string::const_iterator until,
                                  KernelID init = 0 )
    {
        KernelID code = init;
        for (auto it = from; it != until; ++it)
            code = code * StatesN + _char2ID( *it );
        return code;
    }

    static std::string _id2Sequence( KernelID id, const size_t size, std::string &&acc = "" )
    {
        if ( acc.size() == size ) return acc;
        else return _id2Sequence( id / StatesN, size, _id2Char( id % StatesN ) + acc );
    }


private:
    const Order _maxOrder;
    HeteroKernels _kernels;
    size_t _hits;
};


#endif
