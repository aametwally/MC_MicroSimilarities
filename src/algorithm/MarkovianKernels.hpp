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
    static constexpr double PseudoCounts = double( 1 ) / StatesN;

    using Buffer = std::array<double, StatesN>;
    using BufferIterator = typename Buffer::iterator;
    using BufferConstIterator = typename Buffer::const_iterator;

    using Selection = std::unordered_map<Order, std::set<KernelID >>;
    using SelectionFlat = std::unordered_map<Order, std::vector<KernelID >>;
    using SelectionOrdered = std::map<Order, std::set<KernelID>>;

    struct LazySelectionsIntersection
    {
        using ValueType = std::pair<Order, KernelID>;

        struct ConstantIterator
        {
        protected:
            friend class LazySelectionsIntersection;

            static ConstantIterator beginIterator( const LazySelectionsIntersection &lazyInt )
            {
                return ConstantIterator( lazyInt, true );
            }

            static ConstantIterator endIterator( const LazySelectionsIntersection &lazyInt )
            {
                return ConstantIterator( lazyInt, false );
            }

        private:

            using OrderIterator = Selection::const_iterator;
            using IDIterator = std::set<KernelID>::const_iterator;

            std::optional<Order> _currentOrder() const
            {
                if ( _orderIt )
                    return _orderIt.value()->first;
                else return std::nullopt;
            }

            std::optional<KernelID> _currentID() const
            {
                if ( _idIt )
                    return *_idIt.value();
                else return std::nullopt;
            }

            ConstantIterator( const LazySelectionsIntersection &lazyInt, bool begin )
                    : _data( std::cref( lazyInt ))
            {
                if ( begin )
                {
                    _init();
                }
            }

            const Selection &_s1() const
            {
                if ( _data.get()._s1 )
                    return _data.get()._s1.value();
                else return _data.get()._s1Ref->get();
            }

            std::optional<std::reference_wrapper<const Selection>> _s2() const
            {
                if ( _data.get()._s2 )
                    return _data.get()._s2.value();
                else if ( _data.get()._s2Ref )
                    return _data.get()._s2Ref;
                else return std::nullopt;
            }

            std::optional<IDIterator> findFirstIt( Order order,
                                                   const std::set<KernelID> &ids1,
                                                   const std::optional<IDIterator> &start )
            {
                if ( auto s2Opt = _s2(); s2Opt )
                {
                    auto &s2 = s2Opt.value().get();
                    if ( auto s2It = s2.find( order ); s2It != s2.cend())
                    {
                        const auto &ids2 = s2It->second;
                        auto it = std::find_if( start.value_or( ids1.cbegin()),
                                                ids1.cend(),
                                                [&]( const KernelID id ) {
                                                    return ids2.find( id ) != ids2.cend();
                                                } );

                        if ( it != ids1.cend()) return it;
                    }
                }
                return std::nullopt;
            }

            void _init()
            {
                if ( !_s1().empty())
                {
                    _orderIt = _s1().cbegin();

                }
                _next();
            }

            void _next()
            {

                if ( _orderIt )
                {

                    if ( _idIt ) ++_idIt.value();

                    while (_orderIt != _s1().cend())
                    {
                        auto order = _orderIt.value()->first;
                        auto &ids1 = _orderIt.value()->second;
                        if( _idIt = findFirstIt( order , ids1 , _idIt ); _idIt )
                            return;
                        else ++_orderIt.value();
                    }
                    _orderIt = std::nullopt;
                    _idIt = std::nullopt;
                }
            }

        public:
            ConstantIterator( const ConstantIterator & ) = default;

            ConstantIterator &operator=( const ConstantIterator & ) = default;

            ConstantIterator &operator++()
            {
                _next();
                return *this;
            }; //prefix increment
            ConstantIterator operator++( int )
            {
                ConstantIterator tmp( *this );
                _next();
                return tmp;
            }

            ValueType operator*() const
            {
                return std::make_pair( _currentOrder().value(), _currentID().value());
            }

            bool operator==( const ConstantIterator &rhs ) const
            { return _orderIt == rhs._orderIt && _idIt == rhs._idIt; }

            bool operator!=( const ConstantIterator &rhs ) const
            { return _orderIt != rhs._orderIt || _idIt != rhs._idIt; }

        private:

            std::reference_wrapper<const LazySelectionsIntersection> _data;
            std::optional<OrderIterator> _orderIt;
            std::optional<IDIterator> _idIt;
        };

        LazySelectionsIntersection( const Selection &s1, const Selection &s2 )
                : _s1Ref( s1 ), _s2Ref( s2 )
        {}

        LazySelectionsIntersection( const Selection &&s1, const Selection &s2 )
                : _s1( std::move( s1 )), _s2Ref( s2 )
        {}

        LazySelectionsIntersection( const Selection &s1, const Selection &&s2 )
                : _s1Ref( s1 ), _s2( std::move( s2 ))
        {}

        LazySelectionsIntersection( const Selection &&s1, const Selection &&s2 )
                : _s1( std::move( s1 )), _s2( std::move( s2 ))
        {}

        inline ConstantIterator begin() const
        {
            return ConstantIterator::beginIterator( *this );
        }

        inline ConstantIterator cbegin() const
        {
            return ConstantIterator::beginIterator( *this );
        }

        inline ConstantIterator end() const
        {
            return ConstantIterator::endIterator( *this );
        }

        inline ConstantIterator cend() const
        {
            return ConstantIterator::endIterator( *this );
        }

        SelectionOrdered toSelectionOrdered() const
        {
            SelectionOrdered selection;
            for (auto[order, id] : *this)
                selection[order].insert( id );
            return selection;
        }

        static SelectionOrdered toSelectionOrdered( const Selection &s )
        {
            SelectionOrdered selection;
            for (auto &[order, ids] : s)
                for (auto id : ids)
                    selection[order].insert( id );
            return selection;
        }

        static SelectionOrdered toSelectionOrdered( const SelectionFlat &s )
        {
            SelectionOrdered selection;
            for (auto &[order, ids] : s)
                for (auto id : ids)
                    selection[order].insert( id );
            return selection;
        }

        bool equals( const Selection &s ) const
        {
            return toSelectionOrdered() == toSelectionOrdered( s );
        }

        bool equals( const SelectionFlat &s ) const
        {
            auto m1 = toSelectionOrdered();
            auto m2 = toSelectionOrdered( s );
            if ( m1.size() != m2.size()) return false;
            for (auto &[order, ids] : m1)
                if ( ids != m2.at( order ))
                    return false;
            return true;
        }

        bool equals_assert( const SelectionFlat &s ) const
        {
            auto m1 = toSelectionOrdered();
            auto m2 = toSelectionOrdered( s );
            if ( m1.size() != m2.size())
            {
                assert( 0 );
                return false;
            }
            for (auto &[order, ids] : m1)
                if ( ids != m2.at( order ))
                {
                    assert( 0 );
                    return false;
                }
            return true;
        }

        static size_t size( const MarkovianKernels &p )
        {
            return p.kernelsCount();
        }

        static size_t size( const Selection &s1 )
        {
            size_t sum = 0;
            for (auto &[order, ids] : s1)
                sum += ids.size();
            return sum;
        }

        template<typename Set1, typename Set2>
        static LazySelectionsIntersection intersection( Set1 &&s1, Set2 &&s2 )
        {
            auto n1 = size( s1 );
            auto n2 = size( s2 );
            if ( n1 * std::log2( n2 ) < n2 * std::log2( n1 ))
            {
                return LazySelectionsIntersection( std::forward<Set1>( s1 ), std::forward<Set2>( s2 ));
            } else return LazySelectionsIntersection( std::forward<Set2>( s2 ), std::forward<Set1>( s1 ));
        }

    private:
        std::optional<std::reference_wrapper<const Selection>> _s1Ref;
        std::optional<std::reference_wrapper<const Selection>> _s2Ref;
        std::optional<const Selection> _s1;
        std::optional<const Selection> _s2;
    };


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
    explicit MarkovianKernels( Order mnOrder , Order mxOrder ) :
            _order( mnOrder , mxOrder ), _characters( 0 )
    {
        assert( mxOrder >= mnOrder );
    }

    explicit MarkovianKernels( const std::pair<Order,Order> &order ) :
            _order( order ), _characters( 0 )
    {
        assert( maxOrder() >= minOrder() );
    }

    explicit MarkovianKernels( const std::vector<std::string> &sequences, Order mnOrder , Order mxOrder ) :
            _order( mnOrder , mxOrder ), _characters( 0 )
    {
        assert( maxOrder() >= minOrder() );
        train( sequences );
    }

    MarkovianKernels() = delete;

    MarkovianKernels( const MarkovianKernels &mE ) = default;

    MarkovianKernels( MarkovianKernels &&mE ) noexcept
            : _order( mE.order()), _kernels( std::move( mE._kernels )), _characters( mE._characters )
    {

    }

    MarkovianKernels &operator=( const MarkovianKernels &mE )
    {
        assert( _order == mE._order );
        if ( _order != mE._order )
            throw std::runtime_error( "Orders mismatch!" );
        _kernels = mE._kernels;
        _characters = mE._characters;
        return *this;
    }

    MarkovianKernels &operator=( MarkovianKernels &&mE )
    {
        assert( _order == mE._order );
        if ( _order != mE._order )
            throw std::runtime_error( "Orders mismatch!" );
        _kernels = std::move( mE._kernels );
        _characters = mE._characters;
        return *this;
    }

    size_t kernelsCount() const
    {
        size_t sum = 0;
        for (auto &[order, isoKernels] : _kernels)
            sum += isoKernels.size();
        return sum;
    }

    bool contains( Order order ) const
    {
        auto isoKernelsIt = _kernels.find( order );
        return isoKernelsIt != _kernels.cend();
    }

    bool contains( Order order, KernelID id ) const
    {
        if ( auto isoKernelsIt = _kernels.find( order ); isoKernelsIt != _kernels.cend())
        {
            auto kernelIt = isoKernelsIt->second.find( id );
            return kernelIt != isoKernelsIt->second.cend();
        } else return false;
    }

    Selection featureSpace() const noexcept
    {
        Selection features;
        for (auto order = minOrder(); order <= maxOrder(); ++order)
            if ( auto isoKernels = kernels( order ); isoKernels )
            {
                for (auto &[id, histogram] : isoKernels.value().get())
                {
                    features[order].insert( id );
                }
            }
        return features;
    }

    static Selection union_( const Selection &s1, const Selection &s2, Order mnOrder , Order mxOrder )
    {
        Selection _union;
        for (auto order = mnOrder; order <= mxOrder; ++order)
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

    static Selection union_( const std::vector<Selection> &sets, Order mnOrder , Order mxOrder )
    {
        Selection scannedKernels;
        for (const auto &selection : sets)
        {
            scannedKernels = union_( scannedKernels, selection, mnOrder , mxOrder );
        }
        return scannedKernels;
    }

    static SelectionFlat intersection2( const Selection &s1, const Selection &s2 )
    {
        SelectionFlat sInt;
        for (auto &[order, ids1] : s1)
        {
            std::vector<KernelID> intersect;
            if ( auto ids2It = s2.find( order ); ids2It != s2.cend())
            {
                const auto &ids2 = ids2It->second;
                std::set_intersection( ids1.cbegin(), ids1.cend(), ids2.cbegin(), ids2.cend(),
                                       std::back_inserter( intersect ));
            }
            if ( !intersect.empty()) sInt[order] = std::move( intersect );
        }
        return sInt;
    }

    static Selection intersection( Selection &&s1, const Selection &s2 ) noexcept
    {
        for (auto &[order, ids1] : s1)
        {
            std::set<KernelID> intersect;
            if ( auto ids2It = s2.find( order ); ids2It != s2.cend())
            {
                const auto &ids2 = ids2It->second;
                std::set_intersection( ids1.cbegin(), ids1.cend(), ids2.cbegin(), ids2.cend(),
                                       std::inserter( intersect, intersect.end()));
            }
            if ( intersect.empty())
                s1.erase( order );
            else s1[order] = std::move( intersect );
        }
        return s1;
    }

    static Selection intersection( const Selection &s1, const Selection &s2, Order minOrder , Order maxOrder ) noexcept
    {
        Selection _intersection;
        for (auto order = minOrder ; order <= maxOrder ; ++order)
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
    intersection( const std::vector<Selection> sets, Order mnOrder, Order mxOrder,
                  std::optional<double> minCoverage = std::nullopt )
    {
        const size_t k = sets.size();
        if ( minCoverage && minCoverage == 0.0 )
        {
            return union_( sets, mnOrder, mxOrder );
        }
        if ( minCoverage && minCoverage > 0.0 )
        {
            const Selection scannedKernels = union_( sets, mnOrder, mxOrder );
            Selection result;
            for (const auto &[order, ids] : scannedKernels)
            {
                for (auto id : ids)
                {
                    auto shared = std::count_if( std::cbegin( sets ), std::cend( sets ),
                                                 [order, id]( const auto &set ) {
                                                     const auto &isoKernels = set.at( order );
                                                     return isoKernels.find( id ) != isoKernels.cend();
                                                 } );
                    if ( shared >= minCoverage.value() * k )
                    {
                        result[order].insert( id );
                    }
                }
            }
            return result;
        } else
        {
            Selection result = sets.front();
            for (auto i = 1; i < sets.size(); ++i)
            {
                result = intersection( result, sets[i], mnOrder, mxOrder );
            }
            return result;
        }

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

    static Order minOrder( const MarkovianProfiles &profiles )
    {
        return profiles.cbegin()->second.minOrder();
    }

    static Order maxOrder( const HeteroKernelsFeatures &features )
    {
        return std::accumulate( std::cbegin( features ),
                                std::cend( features ), Order( features.cbegin()->first ),
                                []( Order mxOrder, const auto &p ) {
                                    return std::max( mxOrder, p.first );
                                } );
    }

    static Order minOrder( const HeteroKernelsFeatures &features )
    {
        return std::accumulate( std::cbegin( features ),
                                std::cend( features ), Order( features.cbegin()->first ),
                                []( Order minOrder , const auto &p ) {
                                    return std::min( minOrder, p.first );
                                } );
    }

    static Selection
    jointFeatures( const MarkovianProfiles &profiles,
                   const std::unordered_map<Order, std::set<KernelID >> &allFeatures,
                   std::optional<double> minSharedPercentage = std::nullopt )
    {
        const Order mxOrder = maxOrder( profiles );
        const Order mnOrder = minOrder( profiles );

        Selection joint;
        if ( minSharedPercentage )
        {
            assert( minSharedPercentage > 0.0 );

            const size_t minShared = size_t( profiles.size() * minSharedPercentage.value());
            for (auto order = mnOrder; order <= mxOrder; ++order)
                for (const auto id : allFeatures.at( order ))
                {
                    auto shared = std::count_if( std::cbegin( profiles ), std::cend( profiles ),
                                                 [order, id]( const auto &p ) {
                                                     const auto kernel = p.second.kernel( order, id );
                                                     return kernel.has_value();
                                                 } );
                    if ( shared >= minShared )
                        joint[order].insert( id );

                }
        } else
        {
            for (auto order = mnOrder; order <= mxOrder; ++order)
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
    coveredFeatures( const std::map<std::string, std::vector<std::string >> &train, Order minOrder, Order maxOrder )
    {
        std::map<std::string, MarkovianKernels> profiles;
        for (const auto &[label, seqs] : train)
        {
            profiles.emplace( label, MarkovianKernels( minOrder , maxOrder ));
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
        std::map<std::string, MarkovianKernels> filteredProfiles;
        for (auto &[cluster, profile] : profiles)
        {
            if ( auto filtered = filter( std::move( profile ), selection ); filtered )
                filteredProfiles.emplace( cluster, std::move( filtered.value()));
        }
        return filteredProfiles;
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

    static std::optional<MarkovianKernels>
    filter( MarkovianKernels &&other, const Selection &select ) noexcept
    {
        using LazyIntersection = LazySelectionsIntersection;

        HeteroKernels selectedKernels;

        assert( LazyIntersection::intersection( select, other.featureSpace())
                        .equals_assert( intersection2( select, other.featureSpace())));

//        for (auto &[order, ids] : intersection2( select, other.featureSpace()))
//            for( auto id : ids )
//            selectedKernels[order][id] = std::move( other.kernel( order, id ).value());
        for (auto[order, id] : LazyIntersection::intersection( select, other.featureSpace()))
            selectedKernels[order][id] = std::move( other.kernel( order, id ).value());

        if ( selectedKernels.empty())
            return std::nullopt;
        else
        {
            MarkovianKernels filteredProfiles( other._order );
            filteredProfiles._kernels = std::move( selectedKernels );
            filteredProfiles._characters = other._characters;
            return std::make_optional( filteredProfiles );
        }
    }

    void train( const std::vector<std::string> &sequences )
    {
        for (const auto &s : sequences)
            _countInstance( s );

        for (Order order = minOrder(); order <= maxOrder(); ++order)
            for (auto &[id, kernel] : _kernels.at( order ))
                kernel.normalize();
    }

    static std::map<std::string, MarkovianKernels>
    train( const std::map<std::string, std::vector<std::string >> &training,
           Order mnOrder, Order mxOrder ,
           std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
    {
        MarkovianProfiles trainedProfiles;

        for (const auto &[label, sequences] : training)
        {
            MarkovianKernels kernel( mnOrder , mxOrder );
            kernel.train( sequences );

            if ( selection )
            {
                if ( auto profile = filter( std::move( kernel ), selection->get()); profile )
                    trainedProfiles.emplace( label, std::move( profile.value()));

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
        assert( !allHits.empty());
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
        for (Order order = minOrder(); order <= maxOrder(); ++order)
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

    const std::pair<Order,Order> &order() const
    {
        return _order;
    }

    Order minOrder() const
    {
        return _order.first;
    }

    Order maxOrder() const
    {
        return _order.second;
    }

    static constexpr inline KernelID lowerOrderID( KernelID id )
    { return id / StatesN; }


    class KernelsFeaturesByOrder : public Series<double, KernelsFeaturesByOrder>
    {
    public:
        KernelsFeaturesByOrder( const HeteroKernelsFeatures &features,
                                std::pair< Order , Order > range ,
                                Order order,
                                KernelID id )
                : _features( std::cref( features )),
                  _mutables( order, id )
        {}

        inline bool isEmpty() const
        {
            return currentOrder() < _range.first;
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
                return currentOrder() + 1 - _range.first;
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
        std::pair<Order,Order> _range;
        std::pair<Order, KernelID> _mutables;
    };

    template<typename Derived, typename ReturnType>
    class ObjectSeriesByOrder : public Series<ReturnType, Derived, ObjectSeriesByOrder<Derived, ReturnType>>
    {
    public:
        ObjectSeriesByOrder( const MarkovianKernels &kernels,
                             std::pair< Order , Order > range,
                             Order order,
                             KernelID id )
                : _kernels( std::cref( kernels )), _range( range ),
                  _mutables( order, id )
        {}

        inline bool isEmpty() const
        {
            return currentOrder() < _range.first;
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
                return currentOrder() + 1 - _range.first;
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
        std::pair<Order,Order> _range;
        std::pair<Order, KernelID> _mutables;
    };

    class KernelSeriesByOrder : public ObjectSeriesByOrder<KernelSeriesByOrder, std::reference_wrapper<const Kernel >>
    {
    public:
        KernelSeriesByOrder( const MarkovianKernels &kernels,
                             std::pair<Order,Order> range,
                             Order order,
                             KernelID id )
                : ObjectSeriesByOrder<KernelSeriesByOrder, std::reference_wrapper<const Kernel >>( kernels, range , order, id )
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
                             std::pair<Order,Order> range,
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
        return KernelSeriesByOrder( *this, _order , _order.second , id );
    }

    inline KernelSeriesByOrder kernelsByOrder( Order order, KernelID id ) const
    {
        return KernelSeriesByOrder( *this, _order, _order.second ,  id );
    }

    inline ProbabilitisByOrder probabilitisByOrder( Order order, KernelID id ) const
    {
        return ProbabilitisByOrder( *this, _order, _order.second , id );
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
        for (auto order = minOrder(); order <= maxOrder(); ++order)
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
    const std::pair<Order,Order> _order;
    HeteroKernels _kernels;
    size_t _characters;
};


#endif
