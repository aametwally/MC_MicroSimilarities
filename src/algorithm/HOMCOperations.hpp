//
// Created by asem on 09/09/18.
//

#ifndef MARKOVIAN_FEATURES_HOMCOPERATIONS_HPP
#define MARKOVIAN_FEATURES_HOMCOPERATIONS_HPP

#include "HOMCDefs.hpp"
#include "HOMC.hpp"

namespace MC
{

    struct LazySelectionsIntersection
    {
        using ValueType = std::pair<Order, HistogramID>;

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
            using IDIterator = std::set<HistogramID>::const_iterator;

            std::optional<Order> _currentOrder() const
            {
                if ( _orderIt )
                    return _orderIt.value()->first;
                else return std::nullopt;
            }

            std::optional<HistogramID> _currentID() const
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
                                                   const std::set<HistogramID> &ids1,
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
                                                [&]( const HistogramID id ) {
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
                        if ( _idIt = findFirstIt( order, ids1, _idIt ); _idIt )
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
            }

            //prefix increment
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
        explicit KernelIdentifier( Order o, HistogramID i ) : order( o ), id( i )
        {}
        Order order;
        HistogramID id;
    };



    Selection union_( const Selection &s1, const Selection &s2, Order mnOrder, Order mxOrder )
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

    Selection union_( const std::vector<Selection> &sets, Order mnOrder, Order mxOrder )
    {
        Selection scannedKernels;
        for (const auto &selection : sets)
        {
            scannedKernels = union_( scannedKernels, selection, mnOrder, mxOrder );
        }
        return scannedKernels;
    }

    SelectionFlat intersection2( const Selection &s1, const Selection &s2 )
    {
        SelectionFlat sInt;
        for (auto &[order, ids1] : s1)
        {
            std::vector<HistogramID> intersect;
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

    Selection intersection( Selection &&s1, const Selection &s2 ) noexcept
    {
        for (auto &[order, ids1] : s1)
        {
            std::set<HistogramID> intersect;
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

    Selection
    intersection( const Selection &s1, const Selection &s2, Order minOrder, Order maxOrder ) noexcept
    {
        Selection _intersection;
        for (auto order = minOrder; order <= maxOrder; ++order)
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

    Selection
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

    template<typename Grouping>
    struct Ops
    {
        using HOMCP = HOMC< Grouping >;
        using BackboneProfiles = typename HOMCP::BackboneProfiles ;
        using HeteroHistogramsFeatures = typename HOMCP::HeteroHistogramsFeatures ;
        using HeteroHistograms = typename HOMCP::HeteroHistograms ;


        static Order maxOrder( const BackboneProfiles &profiles )
        {
            return profiles.cbegin()->second.maxOrder();
        }

        static Order minOrder( const BackboneProfiles &profiles )
        {
            return profiles.cbegin()->second.minOrder();
        }

        static Order maxOrder( const HeteroHistogramsFeatures &features )
        {
            return std::accumulate( std::cbegin( features ),
                                    std::cend( features ), Order( features.cbegin()->first ),
                                    []( Order mxOrder, const auto &p ) {
                                        return std::max( mxOrder, p.first );
                                    } );
        }

        static Order minOrder( const HeteroHistogramsFeatures &features )
        {
            return std::accumulate( std::cbegin( features ),
                                    std::cend( features ), Order( features.cbegin()->first ),
                                    []( Order minOrder, const auto &p ) {
                                        return std::min( minOrder, p.first );
                                    } );
        }

        static Selection featureSpace( const BackboneProfiles &profiles )
        {
            const Order mxOrder = maxOrder( profiles );
            std::unordered_map<Order, std::set<HistogramID >> allFeatureSpace;

            for (const auto &[cluster, profile] : profiles)
                for (const auto &[order, isoHistograms] : profile.histograms())
                    for (const auto &[id, histogram] : isoHistograms.get())
                        allFeatureSpace[order].insert( id );
            return allFeatureSpace;
        }

        static Selection jointFeatures( const BackboneProfiles &profiles,
                                        const std::unordered_map<Order, std::set<HistogramID >> &allFeatures,
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
                                                         const auto histogram = p.second.histogram( order, id );
                                                         return histogram.has_value();
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
                                                        const auto histogram = p.second.histogram( order, id );
                                                        return histogram.has_value();
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
            std::map<std::string, HOMCP> profiles;
            for (const auto &[label, seqs] : train)
            {
                profiles.emplace( label, HOMC( minOrder, maxOrder ));
                profiles.at( label ).train( seqs );
            }
            auto _allFeatures = featureSpace( profiles );
            auto _jointFeatures = jointFeatures( profiles, _allFeatures );
            return std::make_pair( std::move( _allFeatures ), std::move( _jointFeatures ));
        }

        static std::optional<HOMCP>
        filter( HOMCP &&other, const Selection &select ) noexcept
        {
            using LazyIntersection = LazySelectionsIntersection;

            HeteroHistograms selectedHistograms;

            for (auto[order, id] : LazyIntersection::intersection( select, other.featureSpace()))
                selectedHistograms[order][id] = std::move( other.histogram( order, id ).value());

            if ( selectedHistograms.empty())
                return std::nullopt;
            else
                return HOMCP( other.getOrder() , std::move( selectedHistograms ) );
        }

        static BackboneProfiles
        train( const std::map<std::string, std::vector<std::string >> &training,
               Order mnOrder, Order mxOrder,
               std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
        {
            BackboneProfiles trainedProfiles;

            for (const auto &[label, sequences] : training)
            {
                HOMCP homc( mnOrder, mxOrder );
                homc.train( sequences );

                if ( selection )
                {
                    if ( auto profile = filter( std::move( homc ), selection->get()); profile )
                        trainedProfiles.emplace( label, std::move( profile.value()));

                } else
                {
                    trainedProfiles.emplace( label, std::move( homc ));
                }
            }

            return trainedProfiles;
        }

        static std::pair<Selection, BackboneProfiles>
        filterJointKernels( const std::map<std::string, std::vector<std::string >> &trainingClusters,
                            Order mnOrder, Order mxOrder ,
                            double minSharedPercentage = 0.75 )
        {
            assert( minSharedPercentage > 0 && minSharedPercentage <= 1 );
            return filterJointKernels( train( trainingClusters, mnOrder , mxOrder ), minSharedPercentage );
        }

        static size_t size( const std::unordered_map<Order, std::set<HistogramID >> &features )
        {
            return std::accumulate( std::cbegin( features ), std::cend( features ), size_t( 0 ),
                                    []( size_t s, const auto &p ) {
                                        return s + p.second.size();
                                    } );
        }


        static std::pair<Selection, BackboneProfiles>
        filterJointKernels( BackboneProfiles &&profiles, double minSharedPercentage = 0.75 )
        {
            assert( minSharedPercentage >= 0 && minSharedPercentage <= 1 );
            const size_t k = profiles.size();
            const Order mxOrder = maxOrder( profiles );
            const Order mnOrder = minOrder( profiles );
            auto allFeatures = featureSpace( profiles );
            auto commonFeatures = jointFeatures( profiles, allFeatures, minSharedPercentage );

//        fmt::print("Join/All:{}\n", double( size(commonFeatures)) / size(allFeatures));
            profiles = filter( std::move( profiles ), commonFeatures );

            return std::make_pair( std::move( commonFeatures ), std::move( profiles ));
        }

        static Selection
        withinJointAllUnionKernels( const std::map<std::string, std::vector<std::string>> &trainingClusters,
                                    Order mnOrder,
                                    Order mxOrder,
                                    double withinCoverage = 0.5 )
        {
            const size_t k = trainingClusters.size();
            std::vector<Selection> withinKernels;
            for (const auto &[clusterName, sequences] : trainingClusters)
            {
                std::vector<Selection> clusterJointKernels;
                Selection allKernels;
                for (const auto &s : sequences)
                {
                    HOMCP p( {s}, mnOrder, mxOrder );
                    clusterJointKernels.emplace_back( p.featureSpace());
                }
                withinKernels.emplace_back( intersection( clusterJointKernels, mnOrder, mxOrder, withinCoverage ));
            }
            return union_( withinKernels, mnOrder, mxOrder );
        }

        static std::map<std::string, HOMCP>
        filter( std::map<std::string, HOMCP> &&profiles,
                const Selection &selection )
        {
            std::map<std::string, HOMCP> filteredProfiles;
            for (auto &[cluster, profile] : profiles)
            {
                if ( auto filtered = filter( std::move( profile ), selection ); filtered )
                    filteredProfiles.emplace( cluster, std::move( filtered.value()));
            }
            return filteredProfiles;
        }

        static Selection
        filter( const HeteroHistogramsFeatures &scoredFeatures, double percentage )
        {
            Selection newSelection;
            std::vector<std::pair<KernelIdentifier, double> > flat;
            for (const auto &[order, scores] : scoredFeatures)
                for (auto[id, score] : scores)
                    flat.emplace_back( KernelIdentifier( order, id ), score );

            auto cmp = []( const std::pair<KernelIdentifier, double> &p1,
                           const std::pair<KernelIdentifier, double> &p2 ) {
                return p1.second > p2.second;
            };

            size_t percentileTailIdx = size_t( flat.size() * percentage );
            std::nth_element( flat.begin(), flat.begin() + percentileTailIdx,
                              flat.end(), cmp );

            std::for_each( flat.cbegin(), flat.cbegin() + percentileTailIdx,
                           [&]( const std::pair<KernelIdentifier, double> &p ) {
                               newSelection[p.first.order].insert( p.first.id );
                           } );

            return newSelection;
        }

    };


}

#endif //MARKOVIAN_FEATURES_HOMCOPERATIONS_HPP
