//
// Created by asem on 19/08/18.
//

#ifndef MARKOVIAN_FEATURES_MARKOVIANMODELFEATURES_HPP
#define MARKOVIAN_FEATURES_MARKOVIANMODELFEATURES_HPP

#include "HOMC.hpp"
#include "HOMCOperations.hpp"

namespace MC {
    template<typename Grouping>
    class HOMCFeatures
    {
        using HOMCOps = Ops<Grouping>;
        using HOMCP = HOMC<Grouping>;
        using Histogram= typename HOMCP::Histogram;
        using BackboneProfiles = typename HOMCP::BackboneProfiles ;
        using HeteroHistogramsFeatures = typename HOMCP::HeteroHistogramsFeatures ;
        using HeteroHistograms = typename HOMCP::HeteroHistograms ;

        using DoubleSeries = typename HOMCP::ProbabilitisByOrder;
        using KernelsSeries = typename HOMCP::HistogramSeriesByOrder;
    public:
        static std::map<std::string, std::unordered_map<HistogramID, double >>
        propabilityProduct( const BackboneProfiles &profiles, Order order )
        {
            std::map<std::string, std::unordered_map<HistogramID, double >> p;
            for (auto &[cluster, profile] : profiles)
            {
                auto &_p = p[cluster];
                if ( auto isoKernels = profile.histograms( order ); isoKernels )
                {
                    for (auto &[id, histogram] : isoKernels.value().get())
                        _p[id] = profile.probabilitisByOrder( order, id ).product();
                }
            }
            return p;
        }

        static HeteroHistogramsFeatures
        minMaxScaleByOrder( HeteroHistogramsFeatures &&features )
        {
            for (auto &[order, isohistograms] : features)
            {
                double sum = 0;
                double min = std::numeric_limits<double>::infinity();
                double max = -std::numeric_limits<double>::infinity();

                for (auto &[id, feature] : isohistograms)
                {
                    min = std::min( min, feature );
                    max = std::max( max, feature );
                }

                for (auto &[id, feature] : isohistograms)
                    feature = (feature - min) / (max - min + eps);
            }

            return features;
        }

        static HeteroHistogramsFeatures
        minMaxScale( HeteroHistogramsFeatures &&features )
        {
            double min = inf;
            double max = -inf;
            for (auto &[order, isohistograms] : features)
            {
                for (auto &[id, feature] : isohistograms)
                {
                    if ( !std::isnan( feature ))
                    {
                        min = std::min( min, feature );
                        max = std::max( max, feature );
                    }
                }
            }

            assert( max != min );
            for (auto &[order, isohistograms] : features)
                for (auto &[id, feature] : isohistograms)
                    feature = (feature - min) / (max - min + eps);

            return features;
        }

        static HeteroHistograms
        meanHistograms( const BackboneProfiles &profiles,
                        const std::map<std::string, HeteroHistogramsFeatures> &histogramWeights )
        {
            const Order mnOrder = HOMCOps::minOrder( profiles );
            const Order mxOrder = HOMCOps::maxOrder( profiles );
            HeteroHistograms means;

            for (const auto &[cluster, profile] : profiles)
            {
                const auto &weights = histogramWeights.at( cluster );
                for (auto order = mnOrder; order <= mxOrder; ++order)
                    if ( auto isoKernels = profile.histograms( order ); isoKernels )
                    {
                        for (auto &[id, histogram] : isoKernels.value().get())
                            means[order][id] += (histogram * weights.at( order ).at( id ));
                    }
            }
            return means;
        }

        static HeteroHistograms
        meanHistograms( const BackboneProfiles &profiles )
        {
            return meanHistograms( profiles, histogramWeights( profiles ));
        }


        static std::vector<HeteroHistogramsFeatures>
        histogramWeights( const std::vector<HOMCP> &profiles )
        {
            const Order mnOrder = profiles.front().minOrder();
            const Order mxOrder = profiles.front().maxOrder();

            std::vector<HeteroHistogramsFeatures> weights;
            std::unordered_map<Order, std::set<HistogramID >> scannedIDs;

            for (const auto &[cluster, profile] : profiles)
            {
                weights.emplace_back();
                auto &w = weights.back();
                for (auto order = mnOrder; order <= mxOrder; ++order)
                {
                    if ( auto isoKernels = profile.histograms( order ); isoKernels )
                    {
                        for (auto &[id, histogram] : isoKernels.value().get())
                        {
                            w[order][id] = histogram.hits();
                            scannedIDs[order].insert( id );
                        }
                    }
                }
            }

            for (auto order = mnOrder; order <= mxOrder; ++order)
                for (auto id : scannedIDs.at( order ))
                {
                    double sum = 0;
                    for (auto &w : weights)
                        sum += w[order][id];

                    for (auto &w : weights)
                        w.at( order ).at( id ) /= sum;
                }

            return weights;
        }

        static std::map<std::string, HeteroHistogramsFeatures>
        histogramWeights( const BackboneProfiles &profiles )
        {
            const Order mxOrder = HOMCOps::maxOrder( profiles );
            const Order mnOrder = HOMCOps::minOrder( profiles );

            std::map<std::string, HeteroHistogramsFeatures> weights;
            std::unordered_map<Order, std::set<HistogramID >> scannedIDs;

            for (const auto &[cluster, profile] : profiles)
            {
                auto &w = weights[cluster];
                for (auto order = mnOrder; order <= mxOrder; ++order)
                {
                    if ( auto isoKernels = profile.histograms( order ); isoKernels )
                    {
                        for (auto &[id, histogram] : isoKernels.value().get())
                        {
                            w[order][id] = histogram.hits();
                            scannedIDs[order].insert( id );
                        }
                    }
                }
            }


            for (auto &[order, ids] : scannedIDs)
                for (auto id : ids)
                {
                    double sum = 0;
                    for (auto &[cluster, w] : weights)
                        sum += w[order][id];

                    for (auto &[cluster, w] : weights)
                        w.at( order ).at( id ) /= sum;
                }

            return weights;
        }

        template<typename T>
        using ProfileFeatures = std::unordered_map<Order, std::unordered_map<HistogramID, T >>;

        using ProfileHits = ProfileFeatures<size_t>;

        template<typename It, typename Retriever>
        static ProfileFeatures<size_t>
        summer( It first, It last, Retriever retriever )
        {
            auto k = std::distance( first, last );
            assert( k > 0 );

            ProfileFeatures<size_t> sum;
            for (auto it = first; it != last; ++it)
            {
                const auto &features = retriever( it );
                assert( !features.empty());
                for (auto &[order, isoFeatures] :  features)
                {
                    for (auto &[id, feature] : isoFeatures)
                        sum[order][id] += feature;
                }
            }
            return sum;
        };

        template<typename It, typename Retriever>
        static std::vector<HeteroHistogramsFeatures>
        normalizer( It first, It last, Retriever retriever )
        {
            std::vector<HeteroHistogramsFeatures> normalizedItems;
            ProfileFeatures<size_t> sum = summer( first, last, retriever );
            for (auto it = first; it != last; ++it)
            {
                const auto &features = retriever( it );
                HeteroHistogramsFeatures normalized;
                for (auto &[order, isoFeatures] : features)
                {
                    auto &_notmalized = normalized[order];
                    const auto &_sum = sum.at( order );
                    for (auto &[id, feature] : isoFeatures)
                        _notmalized[id] = double( feature ) / _sum.at( id );
                }
                normalizedItems.emplace_back( std::move( normalized ));
            }
            return normalizedItems;
        };

        static ProfileFeatures<size_t>
        summer( const std::vector<ProfileHits> &counts )
        {
            using It = typename std::vector<ProfileHits>::const_iterator;
            return summer( counts.cbegin(), counts.cend(), []( It it ) -> ProfileHits const & { return *it; } );
        };

        static ProfileFeatures<size_t>
        summer( const std::vector<std::reference_wrapper<const ProfileHits >> &counts )
        {
            using It = typename std::vector<std::reference_wrapper<const ProfileHits> >::const_iterator;
            for (auto &c : counts)
                assert( !c.get().empty());
            return summer( counts.begin(), counts.end(),
                           []( It it ) -> ProfileHits const & {
                               assert( !it->get().empty());
                               return (*it).get();
                           } );
        };

        static std::vector<HeteroHistogramsFeatures>
        normalizer( const std::vector<ProfileHits> &counts )
        {
            using It = typename std::vector<ProfileHits>::const_iterator;
            return normalizer( counts.begin(), counts.end(), []( It it ) -> ProfileHits const & { return *it; } );
        };

        static std::vector<HeteroHistogramsFeatures>
        normalizer( const std::vector<std::reference_wrapper<const ProfileHits >> &counts )
        {
            using It = typename std::vector<std::reference_wrapper<const ProfileHits> >::const_iterator;
            return normalizer( counts.begin(), counts.end(),
                               []( It it ) -> ProfileHits const & { return (*it).get(); } );
        };


        static HeteroHistogramsFeatures
        histogramRelevance_ALL2WITHIN_WEIGHTED( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                                Order minOrder, Order maxOrder,
                                                const Selection &selection )
        {

            std::map<std::string, std::vector<HOMCP >> withinClassProfiles;
            std::map<std::string, std::vector<ProfileHits >> withinClassCounters;
            std::map<std::string, HeteroHistogramsFeatures> withinClassRadius;

            std::vector<std::reference_wrapper<const HOMCP>> populationProfiles;
            std::vector<std::reference_wrapper<const ProfileHits >> populationCounts;


            for (auto &[label, sequences] : trainingItems)
            {
                for (auto &s : sequences)
                {
                    if ( auto profile = HOMCOps::filter( HOMCP( {s}, minOrder, maxOrder ), selection ); profile )
                    {
                        withinClassCounters[label].emplace_back( std::move( profile->hits()));
                        withinClassProfiles[label].emplace_back( std::move( profile.value()));
                    }
                }
            }

            for (const auto &[label, profiles] : withinClassProfiles)
            {
                auto &withinCounters = withinClassCounters.at( label );
                for (auto i = 0; i < profiles.size(); ++i)
                {
                    assert( !withinCounters.at( i ).empty());
                    populationCounts.push_back( std::cref( withinCounters.at( i )));
                    populationProfiles.push_back( std::cref( profiles.at( i )));
                }
            }


            auto populationTotalCounter = summer( populationCounts );
            assert( !populationTotalCounter.empty());

            auto populationRadius = informationRadius_WEIGHTED( std::move( populationProfiles ),
                                                                normalizer( std::move( populationCounts )),
                                                                selection );


            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                auto &_withinClassCounters = withinClassCounters.at( label );
                withinClassRadius.emplace( label, informationRadius_WEIGHTED( profiles,
                                                                              normalizer( _withinClassCounters ),
                                                                              selection ));
            }

            HeteroHistogramsFeatures sumJSD;
            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                auto withinClassTotalCounter = summer( withinClassCounters.at( label ));
                assert( !withinClassTotalCounter.empty());
                const auto &classRadius = withinClassRadius.at( label );

                for (const auto &[order, ids] : selection)
                {
                    auto _wcCounterIt = withinClassTotalCounter.find( order );
                    auto _tCounterIt = populationTotalCounter.find( order );
                    auto _wcRadiusIt = classRadius.find( order );
                    if ( _wcCounterIt != withinClassTotalCounter.cend() &&
                         _tCounterIt != populationTotalCounter.cend() &&
                         _wcRadiusIt != classRadius.cend())
                    {
                        const auto &wcCounter = _wcCounterIt->second;
                        const auto &tCounter = _tCounterIt->second;
                        const auto &wcRadius = _wcRadiusIt->second;
                        auto &_sumJSD = sumJSD[order];
                        for (auto id : ids)
                        {
                            auto wcCounterIt = wcCounter.find( id );
                            auto tCounterIt = tCounter.find( id );
                            auto wcRadiusIt = wcRadius.find( id );

                            if ( wcCounterIt != wcCounter.cend() && tCounterIt != tCounter.cend()
                                 && wcRadiusIt != wcRadius.cend())
                            {

                                double wcIRadius = wcRadiusIt->second;
                                if ( !std::isnan( wcIRadius ))
                                {
                                    _sumJSD[id] += wcIRadius;
                                }
                            }
                        }
                    }

                }
            }

            HeteroHistogramsFeatures relevance;
            for (const auto &[order, ids] : selection)
            {
                auto &_relevance = relevance[order];
                const auto &tRadius = populationRadius.at( order );
                const auto &wcRadius = sumJSD.at( order );
                for (auto id : ids)
                {
                    _relevance[id] = getOr( tRadius, id, nan ) - getOr( wcRadius, id, nan );
                }
            }

            return relevance;
        }

        static HeteroHistogramsFeatures
        histogramRelevance_ALL2WITHIN_UNIFORM( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                               Order minOrder, Order maxOrder,
                                               const Selection &selection )
        {

            std::map<std::string, std::vector<HOMCP >> withinClassProfiles;
            std::map<std::string, HeteroHistogramsFeatures> withinClassRadius;

            std::vector<std::reference_wrapper<const HOMCP>> populationProfiles;

            const auto k = trainingItems.size();


            for (auto &[label, sequences] : trainingItems)
            {
                auto &_profiles = withinClassProfiles[label];
                for (auto &s : sequences)
                    if ( auto profile = HOMCOps::filter( HOMCP( {s}, minOrder, maxOrder ), selection ); profile )
                        _profiles.emplace_back( std::move( profile.value()));
            }

            for (const auto &[label, profiles] : withinClassProfiles)
            {
                for (auto &profile : profiles)
                {
                    populationProfiles.push_back( std::cref( profile ));
                }
            }


            auto populationRadius = informationRadius_UNIFORM( std::move( populationProfiles ),
                                                               selection );


            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                withinClassRadius.emplace( label, informationRadius_UNIFORM( profiles, selection ));
            }

            HeteroHistogramsFeatures sumJSD;
            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                const auto &classRadius = withinClassRadius.at( label );

                for (const auto &[order, ids] : selection)
                {
                    if ( auto _wcRadiusIt = classRadius.find( order ); _wcRadiusIt != classRadius.cend())
                    {
                        const auto &wcRadius = _wcRadiusIt->second;
                        auto &_sumJSD = sumJSD[order];
                        for (auto id : ids)
                        {
                            if ( auto wcRadiusIt = wcRadius.find( id ); wcRadiusIt != wcRadius.cend())
                            {
                                double wcIRadius = wcRadiusIt->second;
                                if ( !std::isnan( wcIRadius ))
                                {
                                    _sumJSD[id] += wcRadiusIt->second;
                                }
                            }

                        }
                    }

                }
            }

            HeteroHistogramsFeatures relevance;
            for (const auto &[order, ids] : selection)
            {
                auto &_relevance = relevance[order];
                auto &tRadius = populationRadius.at( order );
                auto &wcRadius = sumJSD.at( order );
                for (auto id : ids)
                {
                    assert( nan - nan != 0 );
                    _relevance[id] = getOr( tRadius, id, nan ) - getOr( wcRadius, id, nan );
                }
            }

            return relevance;
        }

        static HeteroHistogramsFeatures
        histogramRelevance_ALL2MIN_WEIGHTED( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                             Order minOrder, Order maxOrder,
                                             const Selection &selection )
        {

            std::map<std::string, std::vector<HOMCP >> withinClassProfiles;
            std::map<std::string, std::vector<ProfileHits >> withinClassCounters;
            std::map<std::string, HeteroHistogramsFeatures> withinClassRadius;

            std::vector<std::reference_wrapper<const HOMCP>> populationProfiles;
            std::vector<std::reference_wrapper<const ProfileHits>> populationCounts;

            for (auto &[label, sequences] : trainingItems)
            {
                auto &_counter = withinClassCounters[label];
                auto &_profiles = withinClassProfiles[label];
                for (auto &s : sequences)
                {
                    if ( auto profile = HOMCOps::filter( HOMCP( {s}, minOrder, maxOrder ), selection ); profile )
                    {
                        _counter.emplace_back( std::move( profile->hits()));
                        _profiles.emplace_back( std::move( profile.value()));
                    }
                }
            }

            for (const auto &[label, profiles] : withinClassProfiles)
            {
                auto &withinCounters = withinClassCounters.at( label );
                for (auto i = 0; i < profiles.size(); ++i)
                {
                    populationCounts.push_back( std::cref( withinCounters.at( i )));
                    populationProfiles.push_back( std::cref( profiles.at( i )));
                }
            }

            auto populationTotalCounter = summer( populationCounts );
            auto populationRadius = informationRadius_WEIGHTED( std::move( populationProfiles ),
                                                                normalizer( std::move( populationCounts )),
                                                                selection );


            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                auto &_withinClassCounters = withinClassCounters.at( label );
                withinClassRadius.emplace( label, informationRadius_WEIGHTED( profiles,
                                                                              normalizer( _withinClassCounters ),
                                                                              selection ));
            }

            HeteroHistogramsFeatures minJSD;
            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                auto withinClassTotalCounter = summer( withinClassCounters.at( label ));
                const auto &classRadius = withinClassRadius.at( label );

                for (const auto &[order, ids] : selection)
                {
                    auto _wcCounterIt = withinClassTotalCounter.find( order );
                    auto _tCounterIt = populationTotalCounter.find( order );
                    auto _wcRadiusIt = classRadius.find( order );
                    if ( _wcCounterIt != withinClassTotalCounter.cend() &&
                         _tCounterIt != populationTotalCounter.cend() &&
                         _wcRadiusIt != classRadius.cend())
                    {
                        const auto &wcCounter = _wcCounterIt->second;
                        const auto &tCounter = _tCounterIt->second;
                        const auto &wcRadius = _wcRadiusIt->second;
                        auto &_minJSD = minJSD[order];
                        for (auto id : ids)
                        {
                            auto wcCounterIt = wcCounter.find( id );
                            auto tCounterIt = tCounter.find( id );
                            auto wcRadiusIt = wcRadius.find( id );

                            if ( wcCounterIt != wcCounter.cend() && tCounterIt != tCounter.cend()
                                 && wcRadiusIt != wcRadius.cend())
                            {
                                double wcIRadius = wcRadiusIt->second;
                                if ( !std::isnan( wcIRadius ))
                                {
                                    double w = double( wcCounter.at( id )) / tCounter.at( id );
//                                double w = 1;
                                    if ( auto minJSDIt = _minJSD.find( id ); minJSDIt != _minJSD.end())
                                        minJSDIt->second = std::min( minJSDIt->second, wcRadius.at( id ) / w );
                                    else _minJSD.emplace( id, wcRadius.at( id ) / w );
                                }
                            }
                        }
                    }

                }
            }

            HeteroHistogramsFeatures relevance;
            for (const auto &[order, ids] : selection)
            {
                auto &_relevance = relevance[order];
                const auto &tRadius = populationRadius.at( order );
                const auto &wcRadius = minJSD.at( order );
                for (auto id : ids)
                {
                    _relevance[id] = getOr( tRadius, id, nan ) - getOr( wcRadius, id, nan );
                }
            }
            return relevance;
        }

        static HeteroHistogramsFeatures
        histogramRelevance_MAX2MIN_WEIGHTED( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                             Order minOrder, Order maxOrder,
                                             const Selection &selection )
        {

            std::map<std::string, std::vector< HOMCP >> withinClassProfiles;
            std::map<std::string, std::vector<ProfileHits >> withinClassCounters;
            std::map<std::string, HeteroHistogramsFeatures> withinClassRadius;

            std::vector<std::reference_wrapper<const ProfileHits>> populationCounts;


            for (auto &[label, sequences] : trainingItems)
            {
                auto &_counter = withinClassCounters[label];
                auto &_profiles = withinClassProfiles[label];
                for (auto &s : sequences)
                {
                    if ( auto profile = HOMCOps::filter( HOMCP( {s}, minOrder, maxOrder ), selection ); profile )
                    {
                        _counter.emplace_back( std::move( profile->hits()));
                        _profiles.emplace_back( std::move( profile.value()));
                    }
                }
            }


            for (const auto &[label, withinCounters] : withinClassCounters)
                for (auto &c : withinCounters)
                    populationCounts.push_back( std::cref( c ));

            auto populationTotalCounter = summer( populationCounts );


            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                auto &_withinClassCounters = withinClassCounters.at( label );
                withinClassRadius.emplace( label, informationRadius_WEIGHTED( profiles,
                                                                              normalizer( _withinClassCounters ),
                                                                              selection ));
            }

            HeteroHistogramsFeatures minJSD, maxJSD;
            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                auto withinClassTotalCounter = summer( withinClassCounters.at( label ));
                const auto &classRadius = withinClassRadius.at( label );

                for (const auto &[order, ids] : selection)
                {
                    auto _wcCounterIt = withinClassTotalCounter.find( order );
                    auto _tCounterIt = populationTotalCounter.find( order );
                    auto _wcRadiusIt = classRadius.find( order );
                    if ( _wcCounterIt != withinClassTotalCounter.cend() &&
                         _tCounterIt != populationTotalCounter.cend() &&
                         _wcRadiusIt != classRadius.cend())
                    {
                        const auto &wcCounter = _wcCounterIt->second;
                        const auto &tCounter = _tCounterIt->second;
                        const auto &wcRadius = _wcRadiusIt->second;
                        auto &_minJSD = minJSD[order];
                        auto &_maxJSD = maxJSD[order];

                        for (auto id : ids)
                        {
                            auto wcCounterIt = wcCounter.find( id );
                            auto tCounterIt = tCounter.find( id );
                            auto wcRadiusIt = wcRadius.find( id );
                            if ( wcCounterIt != wcCounter.cend() &&
                                 tCounterIt != tCounter.cend() &&
                                 wcRadiusIt != wcRadius.cend())
                            {
                                double wcIRadius = wcRadiusIt->second;
                                if ( !std::isnan( wcIRadius ))
                                {
                                    double w = double( wcCounterIt->second ) / tCounterIt->second;
                                    if ( auto minJSDIt = _minJSD.find( id ); minJSDIt != _minJSD.end())
                                        minJSDIt->second = std::min( minJSDIt->second, wcRadius.at( id ) / w );
                                    else _minJSD.emplace( id, wcRadius.at( id ) / w );
                                    if ( auto maxJSDIt = _maxJSD.find( id ); maxJSDIt != _maxJSD.end())
                                        maxJSDIt->second = std::max( maxJSDIt->second, wcRadius.at( id ) / w );
                                    else _maxJSD.emplace( id, wcRadius.at( id ) / w );
                                }

                            }

                        }
                    }

                }
            }

            HeteroHistogramsFeatures relevance;
            for (const auto &[order, ids] : selection)
            {
                auto &_relevance = relevance[order];
                const auto &maxRadius = maxJSD.at( order );
                const auto &minRadius = minJSD.at( order );
                for (auto id : ids)
                {
                    _relevance[id] = getOr( maxRadius, id, nan ) - getOr( minRadius, id, nan );
                }
            }

            return relevance;
        }

        static HeteroHistogramsFeatures
        histogramRelevance_MAX2MIN_UNIFORM( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                            Order minOrder, Order maxOrder,
                                            const Selection &selection )
        {

            const auto k = trainingItems.size();
            std::map<std::string, std::vector<HOMCP >> withinClassProfiles;
            std::map<std::string, HeteroHistogramsFeatures> withinClassRadius;

            for (auto &[label, sequences] : trainingItems)
            {
                auto &_profiles = withinClassProfiles[label];
                for (auto &s : sequences)
                    if ( auto profile = HOMCOps::filter( HOMCP( {s}, minOrder, maxOrder ), selection ); profile )
                        _profiles.emplace_back( std::move( profile.value()));
            }

            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                withinClassRadius.emplace( label, informationRadius_UNIFORM( profiles, selection ));
            }

            HeteroHistogramsFeatures minJSD, maxJSD;
            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                const auto &classRadius = withinClassRadius.at( label );

                for (const auto &[order, ids] : selection)
                {
                    if ( auto _wcRadiusIt = classRadius.find( order ); _wcRadiusIt != classRadius.cend())
                    {
                        const auto &wcRadius = _wcRadiusIt->second;
                        auto &_minJSD = minJSD[order];
                        auto &_maxJSD = maxJSD[order];

                        for (auto id : ids)
                        {
                            if ( auto wcIRadiusIt = wcRadius.find( id ); wcIRadiusIt != wcRadius.cend())
                            {
                                double wcIRadius = wcIRadiusIt->second;
                                if ( !std::isnan( wcIRadius ))
                                {
                                    double w = 1.0 / k;
                                    if ( auto minJSDIt = _minJSD.find( id ); minJSDIt != _minJSD.end())
                                        minJSDIt->second = std::min( minJSDIt->second, wcIRadius / w );
                                    else _minJSD.emplace( id, wcIRadius / w );
                                    if ( auto maxJSDIt = _maxJSD.find( id ); maxJSDIt != _maxJSD.end())
                                        maxJSDIt->second = std::max( maxJSDIt->second, wcIRadius / w );
                                    else _maxJSD.emplace( id, wcIRadius / w );
                                }
                            }

                        }
                    }

                }
            }

            HeteroHistogramsFeatures relevance;
            for (const auto &[order, ids] : selection)
            {
                auto &_relevance = relevance[order];
                const auto &maxRadius = maxJSD.at( order );
                const auto &minRadius = minJSD.at( order );
                for (auto id : ids)
                {
                    _relevance[id] = getOr( maxRadius, id, nan ) - getOr( minRadius, id, nan );
                }
            }

            return relevance;
        }

        static HeteroHistogramsFeatures
        histogramRelevance_ALL2MIN_UNIFORM( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                            Order minOrder, Order maxOrder,
                                            const Selection &selection )
        {
            const auto k = trainingItems.size();
            std::map<std::string, std::vector<HOMCP >> withinClassProfiles;
            std::map<std::string, HeteroHistogramsFeatures> withinClassRadius;

            std::vector<std::reference_wrapper<const HOMCP> > populationProfiles;

            for (auto &[label, sequences] : trainingItems)
            {
                auto &_profiles = withinClassProfiles[label];
                for (auto &s : sequences)
                    if ( auto profile = HOMCOps::filter( HOMCP( {s}, minOrder, maxOrder ), selection ); profile )
                        _profiles.emplace_back( std::move( profile.value()));
            }

            for (const auto &[label, profiles] : withinClassProfiles)
            {
                for (auto &profile : profiles)
                {
                    populationProfiles.push_back( std::cref( profile ));
                }
            }

            auto populationRadius = informationRadius_UNIFORM( std::move( populationProfiles ), selection );

            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                withinClassRadius.emplace( label, informationRadius_UNIFORM( profiles,
                                                                             selection ));
            }

            HeteroHistogramsFeatures minJSD;
            for (const auto &[label, profiles] :  withinClassProfiles)
            {
                const auto &classRadius = withinClassRadius.at( label );
                for (const auto &[order, ids] : selection)
                {
                    if ( auto _wcRadiusIt = classRadius.find( order ); _wcRadiusIt != classRadius.cend())
                    {
                        const auto &wcRadius = _wcRadiusIt->second;
                        auto &_minJSD = minJSD[order];
                        for (auto id : ids)
                        {
                            if ( auto wcIRadiusIt = wcRadius.find( id ); wcIRadiusIt != wcRadius.cend())
                            {
                                double wcIRadius = wcIRadiusIt->second;
                                if ( !std::isnan( wcIRadius ))
                                {
                                    double w = double( 1 ) / k;
//                                double w = 1;
                                    if ( auto minJSDIt = _minJSD.find( id ); minJSDIt != _minJSD.end())
                                        minJSDIt->second = std::min( minJSDIt->second, wcIRadius / w );
                                    else _minJSD.emplace( id, wcIRadius / w );
                                }
                            }

                        }
                    }

                }
            }

            HeteroHistogramsFeatures relevance;
            for (const auto &[order, ids] : selection)
            {
                auto &_relevance = relevance[order];
                const auto &tRadius = populationRadius.at( order );
                const auto &wcRadius = minJSD.at( order );
                for (auto id : ids)
                {
                    double val = getOr( tRadius, id, nan ) - getOr( wcRadius, id, nan );
                    if ( std::isnan( val ))
                    {
                        fmt::print( "nan\n" );
                    }
                    _relevance[id] = val;
                }
            }


            return relevance;
        }



        template<typename ProfileForwardIt, typename WeightForwardIt, typename Retriever>
        static HeteroHistogramsFeatures
        informationRadius_WEIGHTED( ProfileForwardIt profileFirstIt, ProfileForwardIt profileLastIt,
                                    WeightForwardIt wFirstIt, WeightForwardIt wLastIt,
                                    const Selection &selection,
                                    Retriever &&retrieve )
        {
            assert( std::distance( profileFirstIt, profileLastIt ) == std::distance( wFirstIt, wLastIt ));
            using LazyIntersection =  LazySelectionsIntersection;

            HeteroHistogramsFeatures meanEntropies;
            HeteroHistograms meanHistogram;
            HeteroHistogramsFeatures radius;
            auto profileIt = profileFirstIt;
            auto weightIt = wFirstIt;
            for (; profileIt != profileLastIt; ++profileIt, ++weightIt)
            {
                const HOMCP &profile = retrieve( profileIt );
                const auto &weights = *weightIt;

                for (auto[order, id] : LazyIntersection::intersection( profile.featureSpace(), selection ))
                {
                    auto &_weights = weights.at( order );
                    auto &_meanEntropies = meanEntropies[order];
                    auto &_meanHistogram = meanHistogram[order];

                    double b = _weights.at( id );
                    const auto &histogram = profile.histogram( order, id ).value().get();
                    _meanEntropies[id] += (histogram.information() * b);
                    _meanHistogram[id] += (histogram * b);
                }
            }

            for (auto &[order, ids] : selection)
            {
                auto isoMeanEntropiesIt = meanEntropies.find( order );
                auto isoMeanHistogramIt = meanHistogram.find( order );
                auto &_radius = radius[order];
                if ( isoMeanEntropiesIt == meanEntropies.cend() || isoMeanHistogramIt == meanHistogram.cend())
                {
                    for (auto id: ids) _radius[id] = nan;
                } else
                {
                    for (auto id: ids)
                    {
                        auto meanEntropyIt = isoMeanEntropiesIt->second.find( id );
                        auto meanHistogramIt = isoMeanHistogramIt->second.find( id );

                        if ( meanEntropyIt != isoMeanEntropiesIt->second.cend() &&
                             meanHistogramIt != isoMeanHistogramIt->second.cend())
                        {
                            const auto &__meanHistogram = meanHistogramIt->second;
                            const auto &meanEntropy = meanEntropyIt->second;
                            _radius[id] = __meanHistogram.information() - meanEntropy;
                        } else _radius[id] = nan;
                    }
                }
            }
            return radius;
        }


        template<typename ProfileForwardIt, typename Retriever>
        static HeteroHistogramsFeatures
        informationRadius_UNIFORM( ProfileForwardIt profileFirstIt, ProfileForwardIt profileLastIt,
                                   const Selection &selection,
                                   Retriever &&retrieve )
        {
            using LazyIntersection = LazySelectionsIntersection;

            const auto k = std::distance( profileFirstIt, profileLastIt );
            HeteroHistogramsFeatures meanEntropies;
            HeteroHistograms meanHistogram;
            HeteroHistogramsFeatures radius;

            for (auto it = profileFirstIt; it != profileLastIt; ++it)
            {
                const HOMCP &profile = retrieve( it );
                for (auto[order, id] : LazyIntersection::intersection( profile.featureSpace(), selection ))
                {

                    double b = double( 1 ) / k;
                    const auto &histogram = profile.histogram( order, id ).value().get();
                    meanEntropies[order][id] += (histogram.information() * b);
                    meanHistogram[order][id] += (histogram * b);
                }
            }

            for (auto &[order, ids] : selection)
            {
                auto isoMeanEntropiesIt = meanEntropies.find( order );
                auto isoMeanHistogramIt = meanHistogram.find( order );
                auto &_radius = radius[order];
                if ( isoMeanEntropiesIt == meanEntropies.cend() || isoMeanHistogramIt == meanHistogram.cend())
                {
                    for (auto id: ids) _radius[id] = nan;
                } else
                {
                    for (auto id: ids)
                    {
                        auto meanEntropyIt = isoMeanEntropiesIt->second.find( id );
                        auto meanHistogramIt = isoMeanHistogramIt->second.find( id );

                        if ( meanEntropyIt != isoMeanEntropiesIt->second.cend() &&
                             meanHistogramIt != isoMeanHistogramIt->second.cend())
                        {
                            const auto &__meanHistogram = meanHistogramIt->second;
                            const auto &meanEntropy = meanEntropyIt->second;
                            _radius[id] = __meanHistogram.information() - meanEntropy;
                        } else _radius[id] = nan;
                    }
                }
            }
            return radius;
        }

        static HeteroHistogramsFeatures
        informationRadius_WEIGHTED( const std::vector<HOMCP> &profiles,
                                    const std::vector<HeteroHistogramsFeatures> &histogramsWeights,
                                    const Selection &selection )
        {
            return informationRadius_WEIGHTED( profiles.cbegin(), profiles.cend(),
                                               histogramsWeights.cbegin(), histogramsWeights.cend(),
                                               selection, []( auto it ) -> const HOMCP & { return *it; } );
        }

        static HeteroHistogramsFeatures
        informationRadius_UNIFORM( const std::vector<HOMCP> &profiles,
                                   const Selection &selection )
        {
            return informationRadius_UNIFORM( profiles.cbegin(), profiles.cend(),
                                              selection, []( auto it ) -> const HOMCP & { return *it; } );
        }

        static HeteroHistogramsFeatures
        informationRadius_WEIGHTED( std::vector<std::reference_wrapper<const HOMCP>> &&profiles,
                                    const std::vector<HeteroHistogramsFeatures> &histogramsWeights,
                                    const Selection &selection )
        {
            using It = typename std::vector<std::reference_wrapper<const HOMCP>>::const_iterator;
            return informationRadius_WEIGHTED( profiles.cbegin(), profiles.cend(),
                                               histogramsWeights.cbegin(), histogramsWeights.cend(),
                                               selection, []( It it ) -> const HOMCP & { return (*it).get(); } );
        }

        static HeteroHistogramsFeatures
        informationRadius_UNIFORM( std::vector<std::reference_wrapper<const HOMCP>> &&profiles,
                                   const Selection &selection )
        {
            using It = typename std::vector<std::reference_wrapper<const HOMCP>>::const_iterator;
            return informationRadius_UNIFORM( profiles.cbegin(), profiles.cend(),
                                              selection, []( It it ) -> const HOMCP & { return (*it).get(); } );
        }

        static HeteroHistogramsFeatures
        informationRadius_WEIGHTED( const BackboneProfiles &profiles,
                                    const std::map<std::string, HeteroHistogramsFeatures> &histogramsWeights )
        {

            const size_t k = profiles.size();

            HeteroHistogramsFeatures meanEntropies;
            HeteroHistograms meanHistogram;
            HeteroHistogramsFeatures radius;

            for (const auto &[cluster, profile] : profiles)
            {
                const auto &weights = histogramsWeights.at( cluster );
                for (const auto[order, isoKernels] : profile.histograms())
                    for (const auto &[id, histogram] : isoKernels.get())
                    {
                        double b = weights.at( order ).at( id );
                        meanEntropies[order][id] += (histogram.information() * b);
                        meanHistogram[order][id] += (histogram * b);
                    }
            }

            for (const auto &[order, isoMeanEntropies] : meanEntropies)
                for (auto &[id, meanEntropy] : isoMeanEntropies)
                {
                    radius[order][id] = meanHistogram.at( order ).at( id ).information() - meanEntropy;
                }
            return radius;
        }

        static HeteroHistogramsFeatures
        informationRadius_UNIFORM( const BackboneProfiles &profiles )
        {
            const Order mxOrder = HOMCOps::maxOrder( profiles );
            const Order mnOrder = HOMCOps::minOrder( profiles );

            const size_t k = profiles.size();

            HeteroHistogramsFeatures meanEntropies;
            HeteroHistograms meanHistogram;
            HeteroHistogramsFeatures radius;

            for (const auto &[cluster, profile] : profiles)
            {
                for (auto &[order, isoKernels] : profile.histograms())

                    for (const auto &[id, histogram] : isoKernels.get())
                    {
                        double b = 1.0 / k;
                        meanEntropies[order][id] += (histogram.information() * b);
                        meanHistogram[order][id] += (histogram * b);
                    }

            }

            for (auto &[order, isoMeanEntropies] : meanEntropies)
                for (auto &[id, meanEntropy] : isoMeanEntropies)
                {
                    radius[order][id] = meanHistogram.at( order ).at( id ).information() - meanEntropy;
                }
            return radius;
        }


    };

}
#endif //MARKOVIAN_FEATURES_MARKOVIANMODELFEATURES_HPP
