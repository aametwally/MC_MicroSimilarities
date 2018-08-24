//
// Created by asem on 19/08/18.
//

#ifndef MARKOVIAN_FEATURES_MARKOVIANMODELFEATURES_HPP
#define MARKOVIAN_FEATURES_MARKOVIANMODELFEATURES_HPP

#include "MarkovianKernels.hpp"


template<typename Grouping>
class MarkovianModelFeatures
{
    using MarkovianProfile = MarkovianKernels<Grouping>;
    using Kernel = typename MarkovianProfile::Kernel;
    using MarkovianProfiles = std::map<std::string, MarkovianProfile>;
    using KernelID = typename MarkovianProfile::KernelID;
    using Order = typename MarkovianProfile::Order;

    using HeteroKernels =  typename MarkovianProfile::HeteroKernels;
    using HeteroKernelsFeatures =  typename MarkovianProfile::HeteroKernelsFeatures;

    using DoubleSeries = typename MarkovianProfile::ProbabilitisByOrder;
    using KernelsSeries = typename MarkovianProfile::KernelSeriesByOrder;
    using Selection = typename MarkovianProfile::Selection;
    using KernelIdentifier = typename MarkovianProfile::KernelIdentifier;

    static constexpr double eps = std::numeric_limits<double>::epsilon();
    static constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    static constexpr double inf = std::numeric_limits<double>::infinity();
    static constexpr Order MinOrder = MarkovianProfile::MinOrder;
    static constexpr size_t StatesN = MarkovianProfile::StatesN;


public:
    static std::map<std::string, std::unordered_map<KernelID, double >>
    propabilityProduct( const MarkovianProfiles &profiles, Order order )
    {
        std::map<std::string, std::unordered_map<KernelID, double >> p;
        for (auto &[cluster, profile] : profiles)
        {
            auto &_p = p[cluster];
            if ( auto isoKernels = profile.kernels( order ); isoKernels )
            {
                for (auto &[id, kernel] : isoKernels.value().get())
                    _p[id] = profile.probabilitisByOrder( order, id ).product();
            }
        }
        return p;
    }

    static HeteroKernelsFeatures
    minMaxScaleByOrder( HeteroKernelsFeatures &&features )
    {
        for (auto &[order, isokernels] : features)
        {
            double sum = 0;
            double min = std::numeric_limits<double>::infinity();
            double max = -std::numeric_limits<double>::infinity();

            for (auto &[id, feature] : isokernels)
            {
                min = std::min( min, feature );
                max = std::max( max, feature );
            }

            for (auto &[id, feature] : isokernels)
                feature = (feature - min) / (max - min + eps);
        }

        return features;
    }

    static HeteroKernelsFeatures
    minMaxScale( HeteroKernelsFeatures &&features )
    {
        double min = inf;
        double max = -inf;
        for (auto &[order, isokernels] : features)
        {
            for (auto &[id, feature] : isokernels)
            {
                if ( !std::isnan( feature ))
                {
                    min = std::min( min, feature );
                    max = std::max( max, feature );
                }
            }
        }

        assert( max != min );
        for (auto &[order, isokernels] : features)
            for (auto &[id, feature] : isokernels)
                feature = (feature - min) / (max - min + eps);

        return features;
    }

    static HeteroKernels
    meanHistograms( const MarkovianProfiles &profiles,
                    const std::map<std::string, HeteroKernelsFeatures> &kernelWeights )
    {
        const Order mxOrder = MarkovianProfile::maxOrder( profiles );
        HeteroKernels means;

        for (const auto &[cluster, profile] : profiles)
        {
            const auto &weights = kernelWeights.at( cluster );
            for (auto order = MinOrder; order <= mxOrder; ++order)
                if ( auto isoKernels = profile.kernels( order ); isoKernels )
                {
                    for (auto &[id, kernel] : isoKernels.value().get())
                        means[order][id] += (kernel * weights.at( order ).at( id ));
                }
        }
        return means;
    }

    static HeteroKernels
    meanHistograms( const MarkovianProfiles &profiles )
    {
        return meanHistograms( profiles, histogramWeights( profiles ));
    }


    static std::vector<HeteroKernelsFeatures>
    histogramWeights( const std::vector<MarkovianProfile> &profiles )
    {
        const Order mxOrder = profiles.front().maxOrder();

        std::vector<HeteroKernelsFeatures> weights;
        std::unordered_map<Order, std::set<KernelID >> scannedIDs;

        for (const auto &[cluster, profile] : profiles)
        {
            weights.emplace_back();
            auto &w = weights.back();
            for (auto order = MinOrder; order <= mxOrder; ++order)
            {
                if ( auto isoKernels = profile.kernels( order ); isoKernels )
                {
                    for (auto &[id, histogram] : isoKernels.value().get())
                    {
                        w[order][id] = histogram.hits();
                        scannedIDs[order].insert( id );
                    }
                }
            }
        }

        for (auto order = MinOrder; order <= mxOrder; ++order)
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

    static std::map<std::string, HeteroKernelsFeatures>
    histogramWeights( const MarkovianProfiles &profiles )
    {
        const Order mxOrder = MarkovianProfile::maxOrder( profiles );

        std::map<std::string, HeteroKernelsFeatures> weights;
        std::unordered_map<Order, std::set<KernelID >> scannedIDs;

        for (const auto &[cluster, profile] : profiles)
        {
            auto &w = weights[cluster];
            for (auto order = MinOrder; order <= mxOrder; ++order)
            {
                if ( auto isoKernels = profile.kernels( order ); isoKernels )
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
    using ProfileFeatures = std::unordered_map<Order, std::unordered_map<KernelID, T >>;

    using ProfileHits = ProfileFeatures<size_t>;

    template<typename Container>
    static ProfileFeatures<size_t>
    summer( const Container &container )
    {
        ProfileFeatures<size_t> sum;
        for (auto &features : container)
        {
            for (auto &[order, isoFeatures] :  features)
            {
                for (auto &[id, feature] : isoFeatures)
                    sum[order][id] += feature;
            }
        }
        return sum;
    };

    template<typename Container>
    static std::vector<HeteroKernelsFeatures>
    normalizer( const Container &container )
    {
        std::vector<HeteroKernelsFeatures> normalizedItems;
        ProfileFeatures<size_t> sum = summer( container );
        for (const ProfileFeatures<size_t> &features : container)
        {
            HeteroKernelsFeatures normalized;
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

    static HeteroKernelsFeatures
    histogramRelevance_ALL2WITHIN_WEIGHTED( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                            Order maxOrder,
                                            const Selection &selection )
    {

        std::map<std::string, std::vector<MarkovianProfile >> withinClassProfiles;
        std::map<std::string, std::vector<ProfileHits >> withinClassCounters;
        std::map<std::string, HeteroKernelsFeatures> withinClassRadius;

        std::vector<MarkovianProfile> populationProfiles;
        std::vector<ProfileHits> populationCounts;


        for (auto &[label, sequences] : trainingItems)
        {
            auto &_counter = withinClassCounters[label];
            auto &_profiles = withinClassProfiles[label];
            for (auto &s : sequences)
            {
                MarkovianProfile profile( {s}, maxOrder );
                profile = MarkovianProfile::filter( std::move( profile ), selection );
                _counter.emplace_back( std::move( profile.hits()));
                _profiles.emplace_back( std::move( profile ));
                populationCounts.push_back( _counter.back());
                populationProfiles.push_back( _profiles.back());
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

        HeteroKernelsFeatures sumJSD;
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

        HeteroKernelsFeatures relevance;
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

    static HeteroKernelsFeatures
    histogramRelevance_ALL2WITHIN_UNIFORM( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                           Order maxOrder,
                                           const Selection &selection )
    {

        std::map<std::string, std::vector<MarkovianProfile >> withinClassProfiles;
        std::map<std::string, HeteroKernelsFeatures> withinClassRadius;

        std::vector<MarkovianProfile> populationProfiles;

        const auto k = trainingItems.size();

        for (auto &[label, sequences] : trainingItems)
        {
            auto &_profiles = withinClassProfiles[label];
            for (auto &s : sequences)
            {
                MarkovianProfile profile( {s}, maxOrder );
                profile = MarkovianProfile::filter( std::move( profile ), selection );
                _profiles.emplace_back( std::move( profile ));
                populationProfiles.push_back( _profiles.back());
            }
        }

        auto populationRadius = informationRadius_UNIFORM( std::move( populationProfiles ),
                                                           selection );


        for (const auto &[label, profiles] :  withinClassProfiles)
        {
            withinClassRadius.emplace( label, informationRadius_UNIFORM( profiles, selection ));
        }

        HeteroKernelsFeatures sumJSD;
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

        HeteroKernelsFeatures relevance;
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

    static HeteroKernelsFeatures
    histogramRelevance_ALL2MIN_WEIGHTED( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                         Order maxOrder,
                                         const Selection &selection )
    {

        std::map<std::string, std::vector<MarkovianProfile >> withinClassProfiles;
        std::map<std::string, std::vector<ProfileHits >> withinClassCounters;
        std::map<std::string, HeteroKernelsFeatures> withinClassRadius;

        std::vector<MarkovianProfile> populationProfiles;
        std::vector<ProfileHits> populationCounts;


        for (auto &[label, sequences] : trainingItems)
        {
            auto &_counter = withinClassCounters[label];
            auto &_profiles = withinClassProfiles[label];
            for (auto &s : sequences)
            {
                MarkovianProfile profile( {s}, maxOrder );
                profile = MarkovianProfile::filter( std::move( profile ), selection );
                _counter.emplace_back( std::move( profile.hits()));
                _profiles.emplace_back( std::move( profile ));
                populationCounts.push_back( _counter.back());
                populationProfiles.push_back( _profiles.back());
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

        HeteroKernelsFeatures minJSD;
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

        HeteroKernelsFeatures relevance;
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

    static HeteroKernelsFeatures
    histogramRelevance_MAX2MIN_WEIGHTED( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                         Order maxOrder,
                                         const Selection &selection )
    {

        std::map<std::string, std::vector<MarkovianProfile >> withinClassProfiles;
        std::map<std::string, std::vector<ProfileHits >> withinClassCounters;
        std::map<std::string, HeteroKernelsFeatures> withinClassRadius;

        std::vector<ProfileHits> populationCounts;


        for (auto &[label, sequences] : trainingItems)
        {
            auto &_counter = withinClassCounters[label];
            auto &_profiles = withinClassProfiles[label];
            for (auto &s : sequences)
            {
                MarkovianProfile profile( {s}, maxOrder );
                profile = MarkovianProfile::filter( std::move( profile ), selection );
                _counter.emplace_back( std::move( profile.hits()));
                _profiles.emplace_back( std::move( profile ));
                populationCounts.push_back( _counter.back());
            }
        }

        auto populationTotalCounter = summer( populationCounts );


        for (const auto &[label, profiles] :  withinClassProfiles)
        {
            auto &_withinClassCounters = withinClassCounters.at( label );
            withinClassRadius.emplace( label, informationRadius_WEIGHTED( profiles,
                                                                          normalizer( _withinClassCounters ),
                                                                          selection ));
        }

        HeteroKernelsFeatures minJSD, maxJSD;
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

        HeteroKernelsFeatures relevance;
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

    static HeteroKernelsFeatures
    histogramRelevance_MAX2MIN_UNIFORM( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                        Order maxOrder,
                                        const Selection &selection )
    {

        const auto k = trainingItems.size();
        std::map<std::string, std::vector<MarkovianProfile >> withinClassProfiles;
        std::map<std::string, HeteroKernelsFeatures> withinClassRadius;


        for (auto &[label, sequences] : trainingItems)
        {
            auto &_profiles = withinClassProfiles[label];
            for (auto &s : sequences)
            {
                MarkovianProfile profile( {s}, maxOrder );
                profile = MarkovianProfile::filter( std::move( profile ), selection );
                _profiles.emplace_back( std::move( profile ));
            }
        }


        for (const auto &[label, profiles] :  withinClassProfiles)
        {
            withinClassRadius.emplace( label, informationRadius_UNIFORM( profiles, selection ));
        }

        HeteroKernelsFeatures minJSD, maxJSD;
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

        HeteroKernelsFeatures relevance;
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

    static HeteroKernelsFeatures
    histogramRelevance_ALL2MIN_UNIFORM( const std::map<std::string, std::vector<std::string >> &trainingItems,
                                        Order maxOrder,
                                        const Selection &selection )
    {
        const auto k = trainingItems.size();
        std::map<std::string, std::vector<MarkovianProfile >> withinClassProfiles;
        std::map<std::string, HeteroKernelsFeatures> withinClassRadius;

        std::vector<MarkovianProfile> populationProfiles;


        for (auto &[label, sequences] : trainingItems)
        {
            auto &_profiles = withinClassProfiles[label];
            for (auto &s : sequences)
            {
                MarkovianProfile profile( {s}, maxOrder );
                profile = MarkovianProfile::filter( std::move( profile ), selection );
                _profiles.emplace_back( profile );
                populationProfiles.emplace_back( std::move( profile ));
            }
        }

        auto populationRadius = informationRadius_UNIFORM( populationProfiles, selection );

        for (const auto &[label, profiles] :  withinClassProfiles)
        {
            withinClassRadius.emplace( label, informationRadius_UNIFORM( profiles,
                                                                         selection ));
        }

        HeteroKernelsFeatures minJSD;
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

        HeteroKernelsFeatures relevance;
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

                }
                _relevance[id] = val;
            }
        }


        return relevance;
    }

    static HeteroKernelsFeatures
    informationRadius_WEIGHTED( const std::vector<MarkovianProfile> &profiles,
                                const std::vector<HeteroKernelsFeatures> &histogramsWeights,
                                const Selection &selection )
    {
        assert( profiles.size() == histogramsWeights.size());

        HeteroKernelsFeatures meanEntropies;
        HeteroKernels meanHistogram;
        HeteroKernelsFeatures radius;

        for (auto i = 0; i < profiles.size(); ++i)
        {
            const MarkovianProfile &profile = profiles[i];
            auto &weights = histogramsWeights[i];
            for (auto &[order, ids] : selection)
            {
                if ( auto isoHistogramsOpt = profile.kernels( order ); isoHistogramsOpt )
                {
                    const auto &isoHistograms = isoHistogramsOpt.value().get();
                    if ( auto weightIt = weights.find( order ); weightIt != weights.cend())
                    {
                        auto &_weights = weightIt->second;
                        auto &_meanEntropies = meanEntropies[order];
                        auto &_meanHistogram = meanHistogram[order];

                        for (auto id : ids)
                        {
                            auto weightIt = _weights.find( id );
                            auto &meanEntropy = _meanEntropies[id];
                            auto &__meanHistogram = _meanHistogram[id];
                            if ( auto histogramIt = isoHistograms.find( id );
                                    histogramIt != isoHistograms.cend() && weightIt != _weights.cend())
                            {
                                double b = weightIt->second;
                                const auto &histogram = histogramIt->second;
                                meanEntropy += (histogram.information() * b);
                                __meanHistogram += (histogram * b);
                            }

                        }

                    }
                }

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

    static HeteroKernelsFeatures
    informationRadius_UNIFORM( const std::vector<MarkovianProfile> &profiles,
                               const Selection &selection )
    {
        const auto k = profiles.size();
        HeteroKernelsFeatures meanEntropies;
        HeteroKernels meanHistogram;
        HeteroKernelsFeatures radius;

        for (auto i = 0; i < profiles.size(); ++i)
        {
            const MarkovianProfile &profile = profiles[i];
            for (auto &[order, ids] : selection)
            {
                if ( auto isoHistogramsOpt = profile.kernels( order ); isoHistogramsOpt )
                {
                    const auto &isoHistograms = isoHistogramsOpt.value().get();
                    auto &_meanEntropies = meanEntropies[order];
                    auto &_meanHistogram = meanHistogram[order];

                    for (auto id : ids)
                    {
                        auto &meanEntropy = _meanEntropies[id];
                        auto &__meanHistogram = _meanHistogram[id];
                        if ( auto histogramIt = isoHistograms.find( id ); histogramIt != isoHistograms.cend())
                        {
                            double b = 1.0 / k;
                            const auto &histogram = histogramIt->second;
                            meanEntropy += (histogram.information() * b);
                            __meanHistogram += (histogram * b);
                        }

                    }
                }
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

    static HeteroKernelsFeatures
    informationRadius_WEIGHTED( const MarkovianProfiles &profiles,
                                const std::map<std::string, HeteroKernelsFeatures> &histogramsWeights )
    {
        const size_t k = profiles.size();

        HeteroKernelsFeatures meanEntropies;
        HeteroKernels meanHistogram;
        HeteroKernelsFeatures radius;

        for (const auto &[cluster, profile] : profiles)
        {
            const auto &weights = histogramsWeights.at( cluster );
            for (const auto[order, isoKernels] : profile.kernels())
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

    static HeteroKernelsFeatures
    informationRadius_UNIFORM( const MarkovianProfiles &profiles )
    {
        const Order mxOrder = MarkovianProfile::maxOrder( profiles );
        const size_t k = profiles.size();

        HeteroKernelsFeatures meanEntropies;
        HeteroKernels meanHistogram;
        HeteroKernelsFeatures radius;

        for (const auto &[cluster, profile] : profiles)
        {
            for (auto &[order, isoKernels] : profile.kernels())

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

    static size_t size( const std::unordered_map<Order, std::set<KernelID >> &features )
    {
        return std::accumulate( std::cbegin( features ), std::cend( features ), size_t( 0 ),
                                []( size_t s, const auto &p ) {
                                    return s + p.second.size();
                                } );
    }

    static std::pair<Selection, MarkovianProfiles>
    filterJointKernels( const std::map<std::string, std::vector<std::string >> &trainingClusters,
                        Order order,
                        double percentage = 0.75 )
    {
        assert( percentage > 0 && percentage <= 1 );
        auto profiles = MarkovianProfile::train( trainingClusters, order );
        return filterJointKernels( std::move( profiles ), percentage );
    }

    static std::pair<Selection, MarkovianProfiles>
    filterJointKernels( MarkovianProfiles &&profiles, double percentage = 0.75 )
    {
        assert( percentage > 0 && percentage <= 1 );
        const size_t k = profiles.size();
        const Order mxOrder = MarkovianProfile::maxOrder( profiles );
        auto allFeatures = MarkovianProfile::featureSpace( profiles );
        auto commonFeatures = MarkovianProfile::jointFeatures( profiles, allFeatures, size_t( percentage * k ));

//        fmt::print("Join/All:{}\n", double( size(commonFeatures)) / size(allFeatures));
        profiles = MarkovianProfile::filter( std::move( profiles ), commonFeatures );

        return std::make_pair( std::move( commonFeatures ),
                               std::move( profiles ));
    }

    static Selection
    withinJointAllUnionKernels( const std::map<std::string, std::vector<std::string>> &trainingClusters,
                                Order order,
                                double withinCoverage = 0.5 )
    {
        assert( withinCoverage > 0 && withinCoverage <= 1 );
        const size_t k = trainingClusters.size();
        std::vector<Selection> withinKernels;
        for (const auto &[clusterName, sequences] : trainingClusters)
        {
            std::vector<Selection> clusterJointKernels;
            Selection allKernels;
            for (const auto &s : sequences)
            {
                MarkovianProfile p( {s}, order );
                clusterJointKernels.emplace_back( p.featureSpace());
            }
            withinKernels.emplace_back( MarkovianProfile::intersection( clusterJointKernels, order, withinCoverage ));
        }
        return MarkovianProfile::union_( withinKernels, order );
    }

    static MarkovianProfiles
    filter( MarkovianProfiles &&profiles, const Selection &selection )
    {
        for (auto &[label, profile] : profiles)
            profile = MarkovianProfile::filter( std::move( profile ), selection );
        return profiles;
    }

    static Selection
    filter( const HeteroKernelsFeatures &scoredFeatures, double percentage )
    {
        Selection newSelection;
        std::vector<std::pair<KernelIdentifier, double> > flat;
        for (const auto &[order, scores] : scoredFeatures)
            for (auto[id, score] : scores)
                flat.emplace_back( KernelIdentifier( order, id ), score );

        auto cmp = []( const std::pair<KernelIdentifier, double> &p1, const std::pair<KernelIdentifier, double> &p2 ) {
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


#endif //MARKOVIAN_FEATURES_MARKOVIANMODELFEATURES_HPP
