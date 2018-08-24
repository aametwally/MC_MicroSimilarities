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

    using Selection = std::unordered_map<Order, std::set<KernelID >>;


    struct KernelIdentifier
    {
        explicit KernelIdentifier( Order o, KernelID i ) : order( o ), id( i )
        {}

        Order order;
        KernelID id;
    };

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


    using MarkovianProfiles = std::map<std::string, MarkovianKernels>;
public:
    explicit MarkovianKernels( Order order ) :
            _maxOrder( order ), _characters( 0 )
    {
        assert( order >= MinOrder );
    }

    explicit MarkovianKernels( const std::vector<std::string> &sequences, Order order ) :
            _maxOrder( order ), _characters( 0 )
    {
        assert( order >= MinOrder );
        train( sequences );
    }

    MarkovianKernels() = delete;

    MarkovianKernels( const MarkovianKernels &mE ) = default;

    MarkovianKernels( MarkovianKernels &&mE ) noexcept
            : _maxOrder( mE.maxOrder()), _kernels( std::move( mE._kernels )), _characters( mE._characters )
    {

    }

    MarkovianKernels &operator=( const MarkovianKernels &mE )
    {
        assert( _maxOrder == mE._maxOrder );
        if ( _maxOrder != mE._maxOrder )
            throw std::runtime_error( "Orders mismatch!" );
        _kernels = mE._kernels;
        _characters = mE._characters;
        return *this;
    }

    MarkovianKernels &operator=( MarkovianKernels &&mE )
    {
        assert( _maxOrder == mE._maxOrder );
        if ( _maxOrder != mE._maxOrder )
            throw std::runtime_error( "Orders mismatch!" );
        _kernels = std::move( mE._kernels );
        _characters = mE._characters;
        return *this;
    }

    Selection featureSpace() const noexcept
    {
        Selection features;
        for (auto order = MinOrder; order <= _maxOrder; ++order)
            if ( auto isoKernels = kernels( order ); isoKernels )
            {
                for (auto &[id, histogram] : isoKernels.value().get())
                {
                    features[order].insert( id );
                }
            }
        return features;
    }

    static Selection union_( const Selection &s1, const Selection &s2, Order mxOrder )
    {
        Selection _union;
        for (Order order = MinOrder; order <= mxOrder; ++order)
        {
            auto ids1It = s1.find( order );
            auto ids2It = s2.find( order );
            if ( ids1It != s1.cend() || ids2It != s2.cend())
            {
                auto &result = _union[order];
                if ( ids1It != s1.cend() && ids2It != s2.cend())
                    std::set_union( ids1It->second.cbegin(), ids1It->second.cend(),
                                    ids2It->second.cbegin(), ids2It->second.cend(),
                                    std::inserter( result, result.begin()));
                else if ( ids1It != s1.cend())
                    result = ids1It->second;
                else
                    result = ids2It->second;
            }
        }

        return _union;
    }

    static Selection union_( const std::vector<Selection> &sets, Order mxOrder )
    {
        Selection scannedKernels;
        for (const auto &selection : sets)
        {
            scannedKernels = union_( scannedKernels, selection, mxOrder );
        }
        return scannedKernels;
    }

    static Selection intersection( const Selection &s1, const Selection &s2, Order mxOrder )
    {
        Selection _intersection;
        for (Order order = MinOrder; order <= mxOrder; ++order)
        {
            try
            {
                auto &ids1 = s1.at( order );
                auto &ids2 = s2.at( order );
                auto &result = _intersection[order];
                std::set_intersection( ids1.cbegin(), ids1.cend(), ids2.cbegin(), ids2.cend(),
                                       std::inserter( result, result.end()));
            } catch (const std::out_of_range &)
            {}
        }
        return _intersection;
    }

    static Selection
    intersection( const std::vector<Selection> sets, Order mxOrder, std::optional<double> coverage = std::nullopt )
    {
        const size_t k = sets.size();
        Selection result;
        if ( coverage )
        {
            Selection scannedKernels;
            for (const auto &selection : sets)
            {
                scannedKernels = union_( scannedKernels, selection, mxOrder );
            }

            for (const auto &[order, ids] : scannedKernels)
            {
                for (auto id : ids)
                {
                    auto shared = std::count_if( std::cbegin( sets ), std::cend( sets ),
                                                 [order, id]( const auto &set ) {
                                                     const auto &isoKernels = set.at( order );
                                                     return isoKernels.find( id ) != isoKernels.cend();
                                                 } );
                    if ( shared >= coverage.value() * k )
                    {
                        result[order].insert( id );
                    }
                }
            }

        } else
        {
            result = sets.front();
            for (auto i = 1; i < sets.size(); ++i)
            {
                result = intersection( result, sets[i], mxOrder );
            }
        }

        return result;
    }

    static Selection
    featureSpace( const MarkovianProfiles &profiles )
    {
        const Order mxOrder = maxOrder( profiles );
        std::unordered_map<Order, std::set<KernelID >> allFeatureSpace;

        for (const auto &[cluster, profile] : profiles)
            for (const auto &[order, isoHistograms] : profile.kernels())
                for (const auto &[id, histogram] : isoHistograms.get())
                    allFeatureSpace[order].insert( id );
        return allFeatureSpace;
    }

    static Order maxOrder( const MarkovianProfiles &profiles )
    {
        return profiles.cbegin()->second.maxOrder();
    }

    static Order maxOrder( const HeteroKernelsFeatures &features )
    {
        return std::accumulate( std::cbegin( features ),
                                std::cend( features ), Order( features.cbegin()->first ),
                                []( Order mxOrder, const auto &p ) {
                                    return std::max( mxOrder, p.first );
                                } );
    }

    static Selection
    jointFeatures( const MarkovianProfiles &profiles,
                   const std::unordered_map<Order, std::set<KernelID >> &allFeatures,
                   std::optional<size_t> minShared = std::nullopt )
    {
        const Order mxOrder = maxOrder( profiles );
        Selection joint;

        if ( minShared )
        {
            for (auto order = MinOrder; order <= mxOrder; ++order)
                for (const auto id : allFeatures.at( order ))
                {
                    auto shared = std::count_if( std::cbegin( profiles ), std::cend( profiles ),
                                                 [order, id]( const auto &p ) {
                                                     const auto kernel = p.second.kernel( order, id );
                                                     return kernel.has_value();
                                                 } );
                    if ( shared >= minShared.value())
                        joint[order].insert( id );

                }
        } else
        {
            for (auto order = MinOrder; order <= mxOrder; ++order)
                for (const auto id : allFeatures.at( order ))
                {
                    bool isJoint = std::all_of( std::cbegin( profiles ), std::cend( profiles ),
                                                [order, id]( const auto &p ) {
                                                    const auto kernel = p.second.kernel( order, id );
                                                    return kernel.has_value();
                                                } );
                    if ( isJoint )
                        joint[order].insert( id );

                }
        }


        return joint;
    }

    static std::pair<Selection, Selection>
    coveredFeatures( const std::map<std::string, std::vector<std::string >> &train, Order maxOrder )
    {
        std::map<std::string, MarkovianKernels> profiles;
        for (const auto &[label, seqs] : train)
        {
            profiles.emplace( label, MarkovianKernels( maxOrder ));
            profiles.at( label ).train( seqs );
        }
        auto _allFeatures = featureSpace( profiles );
        auto _jointFeatures = jointFeatures( profiles, _allFeatures );
        return std::make_pair( std::move( _allFeatures ), std::move( _jointFeatures ));
    }

    static std::map<std::string, MarkovianKernels>
    filter( std::map<std::string, MarkovianKernels> &&profiles,
            const Selection &selection )
    {
        for (auto &[cluster, profile] : profiles)
            profile = filter( std::move( profile ), selection );
        return profiles;
    }

    std::vector<double> extractFlatFeatureVector(
            const Selection &select,
            double missingVals = 0 ) const noexcept
    {
        std::vector<double> features;

        features.reserve(
                std::accumulate( std::cbegin( select ), std::cend( select ), size_t( 0 ),
                                 [&]( size_t acc, const auto &pair ) {
                                     return acc + pair.second.size() * StatesN;
                                 } ));

        for (auto &[order, ids] : select)
        {
            auto &isoKernels = kernels( order );
            for (auto id : ids)
            {
                if ( auto kernelIt = isoKernels.find( id ); kernelIt != isoKernels.cend())
                    features.insert( std::end( features ), std::cbegin( *kernelIt ), std::cend( *kernelIt ));
                else
                    features.insert( std::end( features ), StatesN, missingVals );

            }
        }
        return features;
    }

    std::unordered_map<size_t, double>
    extractSparsedFlatFeatureVector(
            const Selection &select ) const noexcept
    {
        std::unordered_map<size_t, double> features;


        for (auto &[order, ids] : select)
        {
            auto &isoKernels = kernels( order );
            for (auto id : ids)
            {
                if ( auto kernelIt = isoKernels.find( id ); kernelIt != isoKernels.cend())
                {
                    size_t offset = order * id * StatesN;
                    for (auto i = 0; i < StatesN; ++i)
                        features[offset + i] = (*kernelIt)[i];
                }
            }
        }
        return features;
    }

    static MarkovianKernels filter( MarkovianKernels &&other,
                                    const Selection &select ) noexcept
    {

        MarkovianKernels filteredProfiles( other._maxOrder );
        HeteroKernels selectedKernels;

        for (auto &[order, ids] : select)
        {
            if ( auto isoKernelsOpt = other.kernels( order ); isoKernelsOpt )
            {
                auto &isoKernels = isoKernelsOpt.value().get();
                auto &_selectedKernels = selectedKernels[order];
                if ( ids.size() < isoKernels.size() * std::log2( ids.size()))
                {
                    for (auto id : ids)
                    {
                        if ( auto selectedIt = isoKernels.find( id ); selectedIt != isoKernels.end())
                            _selectedKernels[id] = std::move( selectedIt->second );

                    }
                } else
                {
                    for (auto &[id, kernel] : isoKernels)
                    {
                        if ( ids.find( id ) != ids.cend())
                            _selectedKernels[id] = std::move( kernel );
                    }

                }
            }

        }
        filteredProfiles._kernels = std::move( selectedKernels );
        filteredProfiles._characters = other._characters;

        return filteredProfiles;
    }

    void train( const std::vector<std::string> &sequences )
    {
        for (const auto &s : sequences)
            _countInstance( s );

        for (Order order = MinOrder; order <= _maxOrder; ++order)
            for (auto &[id, kernel] : _kernels.at( order ))
                kernel.normalize();
    }

    static std::map<std::string, MarkovianKernels>
    train( const std::map<std::string, std::vector<std::string >> &training,
           Order markovianOrder,
           std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
    {
        MarkovianProfiles trainedProfiles;

        for (const auto &[label, sequences] : training)
        {
            MarkovianKernels kernel( markovianOrder );
            kernel.train( sequences );

            if ( selection )
            {
                trainedProfiles.emplace( label, filter( std::move( kernel ), selection.value().get()));

            } else
            {
                trainedProfiles.emplace( label, std::move( kernel ));
            }
        }

        return trainedProfiles;
    }

    std::unordered_map<Order, std::unordered_map<KernelID, size_t >> hits() const
    {
        std::unordered_map<Order, std::unordered_map<KernelID, size_t >> allHits;
        for (auto &[order, isoKernels] : _kernels)
            for (auto &[id, kernel] : isoKernels)
                allHits[order][id] = kernel.hits();
        return allHits;
    }

    size_t hits( Order order ) const
    {
        return std::accumulate( std::cbegin( _kernels.at( order )), std::cend( _kernels.at( order )),
                                size_t( 0 ), []( size_t acc, const auto &p ) {
                    return acc + p.second.hits();
                } );
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

    std::vector<std::pair<Order, std::reference_wrapper<const IsoKernels >>> kernels() const
    {
        std::vector<std::pair<Order, std::reference_wrapper<const IsoKernels >>> kernelsRef;
        for (Order order = MinOrder; order <= maxOrder(); ++order)
        {
            auto isoKernels = kernels( order );
            if ( isoKernels ) kernelsRef.emplace_back( order, isoKernels.value());
        }
        return kernelsRef;
    }

    std::optional<std::reference_wrapper<const IsoKernels>> kernels( Order order ) const
    {
        if ( auto kernelsIt = _kernels.find( order ); kernelsIt != _kernels.cend())
            return std::cref( kernelsIt->second );
        return std::nullopt;
    }

    std::optional<std::reference_wrapper<const Kernel>> kernel( Order order, KernelID id ) const
    {
        if ( auto kernelsOpt = kernels( order ); kernelsOpt )
            if ( auto kernelIt = kernelsOpt.value().get().find( id ); kernelIt != kernelsOpt.value().get().cend())
                return std::cref( kernelIt->second );
        return std::nullopt;
    }

    Order maxOrder() const
    {
        return _maxOrder;
    }

    static constexpr inline KernelID lowerOrderID( KernelID id )
    { return id / StatesN; }


    class KernelsFeaturesByOrder : public Series<double, KernelsFeaturesByOrder>
    {
    public:
        KernelsFeaturesByOrder( const HeteroKernelsFeatures &features,
                                Order order,
                                KernelID id )
                : _features( std::cref( features )),
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

        virtual std::optional<double> currentTerm() const
        {
            if ( !this->isEmpty())
            {
                try
                {
                    auto order = this->currentOrder();
                    auto id = this->currentID();
                    return this->_features.get().at( order ).at( id );
                } catch (const std::out_of_range &)
                {}
            }
            return std::nullopt;
        }

    protected:
        std::reference_wrapper<const HeteroKernelsFeatures> _features;
        std::pair<Order, KernelID> _mutables;
    };

    template<typename Derived, typename ReturnType>
    class ObjectSeriesByOrder : public Series<ReturnType, Derived, ObjectSeriesByOrder<Derived, ReturnType>>
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

        virtual std::optional<ReturnType> currentTerm() const noexcept = 0;

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

        std::optional<std::reference_wrapper<const Kernel >> currentTerm() const noexcept override
        {
            if ( !this->isEmpty())
            {
                auto order = this->_mutables.first;
                auto id = this->_mutables.second;
                return this->_kernels.get().kernel( order, id );
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

        std::optional<double> currentTerm() const noexcept override
        {
            if ( !this->isEmpty())
            {

                auto order = this->currentOrder();
                auto id = this->currentID();
                const auto &kernels = this->_kernels.get();
                const auto &isoKernels = kernels.kernels( order );
                if ( auto kernelIt = isoKernels.find( id ); kernelIt != isoKernels.cend())
                    return double( kernelIt->second.hits()) / kernels.hits( order );

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

    inline size_t totalCharacters() const noexcept
    {
        return _characters;
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
        _characters += sequence.length();
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
    size_t _characters;
};


#endif
