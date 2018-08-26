//
// Created by asem on 09/08/18.
//

#ifndef MARKOVIAN_FEATURES_CONFIGUREDPIPELINE_HPP
#define MARKOVIAN_FEATURES_CONFIGUREDPIPELINE_HPP

#include <numeric>

#include "common.hpp"
#include "VariantGenerator.hpp"
#include "LabeledEntry.hpp"
#include "Timers.hpp"
#include "ConfusionMatrix.hpp"
#include "CrossValidationStatistics.hpp"
#include "crossvalidation.hpp"
#include "MarkovianKernels.hpp"
#include "MarkovianModelFeatures.hpp"

struct Voting
{
};
struct Accumulative
{
};
enum class StrategyEnum
{
    Voting,
    Accumulative
};
static const std::map<std::string, StrategyEnum> ClassificationStrategyLabel = {
        {"voting", StrategyEnum::Voting},
        {"acc",    StrategyEnum::Accumulative}
};

template<typename Grouping, typename Criteria, typename Strategy>
class ConfiguredPipeline
{
public:

private:

    using PriorityQueue = typename MatchSet<Score>::Queue;
    using MF =  MarkovianModelFeatures<Grouping>;
    using MP = MarkovianKernels<Grouping>;

    using Kernel = typename MP::Kernel;
    using MarkovianProfiles = std::map<std::string, MP>;
    using KernelID = typename MP::KernelID;
    using Order = typename MP::Order;

    using HeteroKernels =  typename MP::HeteroKernels;
    using HeteroKernelsFeatures =  typename MP::HeteroKernelsFeatures;

    using DoubleSeries = typename MP::ProbabilitisByOrder;
    using KernelsSeries = typename MP::KernelSeriesByOrder;
    using Selection = typename MP::Selection;

    static constexpr Order MinOrder = MP::MinOrder;
    static constexpr size_t StatesN = MP::StatesN;
    static constexpr double eps = std::numeric_limits<double>::epsilon();
    static constexpr const char *LOADING = "loading";
    static constexpr const char *PREPROCESSING = "preprocessing";
    static constexpr const char *TRAINING = "training";
    static constexpr const char *CLASSIFICATION = "classification";

    using LeaderBoard = typename std::conditional<std::is_same<Voting, Strategy>::value, ClassificationCandidates<Score>, ClassificationCandidates<Criteria> >::type;

public:

    static std::vector<LabeledEntry>
    reducedAlphabetEntries( const std::vector<LabeledEntry> &entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( entries );
    }

    static std::vector<LeaderBoard>
    classify_VALIDATION(
            const std::vector<std::string> &queries,
            const std::vector<std::string> &trueLabels,
            MarkovianProfiles &&targets,
            std::map<std::string, std::vector<std::string >> &&trainingClusters,
            const Selection &selection,
            Order order )
    {
        assert( queries.size() == trueLabels.size());
        const Order mxOrder = MP::maxOrder( targets );

        auto results = predict( queries, targets, std::move( trainingClusters ), selection, order );
        assert( results.size() == queries.size());
        std::vector<LeaderBoard> classifications;
        for (auto i = 0; i < queries.size(); ++i)
        {
            classifications.emplace_back( trueLabels.at( i ),
                                          results.at( i ));
        }
        return classifications;
    }


    static std::vector<PriorityQueue>
    predict_VOTING(
            const std::vector<std::string> &queries,
            const MarkovianProfiles &targets,
            std::map<std::string, std::vector<std::string >> &&kernelRelevance,
            const Selection &selection,
            Order mxOrder )
    {
        std::vector<PriorityQueue> results;
        for (auto &seq : queries)
        {
            using FeatureSeries = typename MP::KernelsFeaturesByOrder;
            std::map<std::string, double> accumulator;

            MP query( mxOrder );
            query.train( {seq} );
//            query = MP::filter( std::move( query ), selection );
            for (const auto &[order, isoKernels] : query.kernels())
                for (const auto &[id, k1] : isoKernels.get())
                {
                    const auto p1 = query.probabilitisByOrder( mxOrder, id );
                    const auto s1 = query.kernelsByOrder( mxOrder, id );
                    PriorityQueue pq( targets.size());

                    for (const auto &[clusterName, profile] : targets)
                    {
                        auto s2 = profile.kernelsByOrder( mxOrder, id );


                        auto val = KernelsSeries::sum( s1, s2,
                                                       [=]( const Kernel &t1, const Kernel &t2 ) {
                                                           return Criteria::measure( t1, t2 );
                                                       } );
                        pq.emplace( clusterName, val );

                    }
                    pq.forTopK( 1, [&]( const auto &candidate ) {
                        std::string label = candidate.getLabel();
                        FeatureSeries relevance( kernelRelevance, mxOrder, id );
//                    FeatureSeries radius( histogramsRadius, mxOrder, id );
//                    double val = radius.dot( relevance );
//                    double val = radius.product()  relevance.product();
//                    double val = radius.sum() + p1.sum();
                        accumulator[label] += p1.sum();
                    } );

                }
            std::pair<std::string, double> maxVotes = *accumulator.begin();
            PriorityQueue vPQ( targets.size());
            for (const auto &[id, votes]: accumulator)
                vPQ.emplace( id, votes );
            results.emplace_back( std::move( vPQ ));
        }
        return results;

    }

    static std::vector<PriorityQueue>
    predict_VOTING2(
            const std::vector<std::string> &queries,
            const MarkovianProfiles &targets,
            std::map<std::string, std::vector<std::string >> &&trainingClusters,
            const Selection &selection,
            Order mxOrder )
    {
        const auto relevance =
                MF::minMaxScale( MF::histogramRelevance_ALL2WITHIN_WEIGHTED( trainingClusters, mxOrder, selection ));
        std::vector<PriorityQueue> results;
        for (auto &seq : queries)
        {
            std::map<std::string, double> voter;

            if ( auto query = MP::filter( MP( {seq}, mxOrder ), selection ); query )
                for (const auto &[order, isoKernels] : query->kernels())
                {
                    for (const auto &[id, k1] : isoKernels.get())
                    {
                        PriorityQueue pq( targets.size());
                        for (const auto &[clusterName, profile] : targets)
                        {
                            if ( auto k2Opt = profile.kernel( order, id ); k2Opt )
                            {
                                auto val = Criteria::measure( k1, k2Opt.value().get());
                                pq.emplace( clusterName, val );
                            }
                        }
                        double score = getOr( relevance, order, id, double( 0 ));

                        pq.forTopK( targets.size(), [&]( const auto &candidate, size_t index ) {
                            std::string label = candidate.getLabel();
                            voter[label] += (1 + score) / (index + 1);
                        } );
                    }
                }

            std::pair<std::string, double> maxVotes = *voter.begin();
            PriorityQueue vPQ( targets.size());
            for (const auto &[id, votes]: voter)
                vPQ.emplace( id, votes );
            results.emplace_back( std::move( vPQ ));
        }
        return results;
    }

    static std::vector<PriorityQueue>
    predict_ACCUMULATIVE( const std::vector<std::string> &queries,
                          const MarkovianProfiles &targets,
                          std::map<std::string, std::vector<std::string >> &&trainingClusters,
                          const Selection &selection,
                          Order mxOrder )
    {
        auto relevance =
                MF::minMaxScaleByOrder( MF::histogramRelevance_MAX2MIN_WEIGHTED( trainingClusters,
                                                                                 mxOrder, selection ));

        std::vector<PriorityQueue> results;
        for (auto &seq : queries)
        {
            PriorityQueue matchSet( targets.size());

            auto queryOpt = MP::filter( MP( {seq}, mxOrder ), selection );
            if ( queryOpt )
            {
                for (const auto &[clusterId, profile] : targets)
                {
                    double sum = 0;
                    for (const auto &[order, isoKernels] : queryOpt->kernels())
                        for (const auto &[id, kernel1] : isoKernels.get())
                        {
                            double score = getOr( relevance, order, id, double( 0 ));
                            auto k2 = profile.kernel( order, id );
                            if ( k2 )
                            {
                                sum += Criteria::measure( kernel1, k2.value().get()) + score;
                            }
                        }
                    matchSet.emplace( clusterId, sum  );
                }
                results.emplace_back( std::move( matchSet ));
            }

        }

        return results;
    }

    static std::vector<PriorityQueue>
    predict( const std::vector<std::string> &queries,
             const MarkovianProfiles &targets,
             std::map<std::string, std::vector<std::string >> &&trainingClusters,
             const Selection &selection,
             Order order )
    {
        if ( std::is_same<Strategy, Voting>::value )
            return predict_VOTING2( queries, targets, std::move( trainingClusters ), selection, order );
        else if ( std::is_same<Strategy, Accumulative>::value )
            return predict_ACCUMULATIVE( queries, targets, std::move( trainingClusters ), selection, order );
        else throw std::runtime_error( "Undefined Strategy!" );
    }

    static std::pair<Selection, MarkovianProfiles>
    featureSelection( const std::map<std::string, std::vector<std::string> > &trainingClusters,
                      Order order )
    {
        auto selection = MF::withinJointAllUnionKernels( trainingClusters, order, 0.05 );
//        auto scoredFeatures = MF::histogramRelevance_ALL2WITHIN_UNIFORM( trainingClusters, order, selection );
//        auto scoredSelection = MF::filter( scoredFeatures, 0.75 );
        auto trainedProfiles = MP::train( std::move( trainingClusters ), order , selection );
        return std::make_pair( std::move( selection ), std::move( trainedProfiles ));
    }

    void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries,
                                 Order order,
                                 size_t k )
    {
        std::set<std::string> labels;
        for (const auto &entry : entries)
            labels.insert( entry.getLabel());

        using Folds = std::vector<std::vector<std::pair<std::string, std::string >>>;

        const Folds folds = Timers::reported_invoke_s( [&]() {
            fmt::print( "[All Sequences:{}]\n", entries.size());
            entries = reducedAlphabetEntries( entries );
            auto groupedEntries = LabeledEntry::groupSequencesByLabels( std::move( entries ));

            auto labels = keys( groupedEntries );
            for (auto &l : labels) l = fmt::format( "{}({})", l, groupedEntries.at( l ).size());

            fmt::print( "[Clusters:{}][{}]\n",
                        groupedEntries.size(),
                        io::join( labels, "|" ));

            return kFoldStratifiedSplit( std::move( groupedEntries ), k );
        }, PREPROCESSING );


        auto extractTest = []( const std::vector<std::pair<std::string, std::string >> &items ) {
            std::vector<std::string> sequences, labels;
            for (const auto item : items)
            {
                labels.push_back( item.first );
                sequences.push_back( item.second );
            }
            return std::make_pair( sequences, labels );
        };

        CrossValidationStatistics validation( k, labels );
        std::unordered_map<long, size_t> histogram;

        for (auto i = 0; i < k; ++i)
        {
            auto trainingClusters = joinFoldsExceptK( folds, i );
            auto[test, tLabels] = extractTest( folds.at( i ));
            auto trainedProfiles = MP::train( trainingClusters, order );
            auto[selection, filteredProfiles] = featureSelection( trainingClusters, order );
            auto classificationResults = classify_VALIDATION( test, tLabels, std::move( filteredProfiles ),
                                                              std::move( trainingClusters ),
                                                              selection, order );

            for (const auto &classification : classificationResults)
            {
                ++histogram[classification.trueClusterRank()];
                validation.countInstance( i, classification.bestMatch(), classification.trueCluster());
            }
        }

        validation.printReport();

        fmt::print( "True Classification Histogram:\n" );

        for (auto &[k, v] : histogram)
        {
            if ( k == -1 )
                fmt::print( "{:<20}Count:{}\n", "Unclassified", v );
            else
                fmt::print( "{:<20}Count:{}\n", fmt::format( "Rank:{}", k ), v );
        }
    }

};

template<typename...>
struct StrategiesList
{
};
using SupportedStrategies = StrategiesList<Voting, Accumulative>;


using PipelineVariant = MakeVariantType<ConfiguredPipeline,
        SupportedAAGrouping,
        SupportedCriteria,
        SupportedStrategies>;

template<typename AAGrouping, typename Criteria>
PipelineVariant getConfiguredPipeline( StrategyEnum strategy )
{
    switch (strategy)
    {
        case StrategyEnum::Accumulative :
            return ConfiguredPipeline<AAGrouping, Criteria, Accumulative>();
        case StrategyEnum::Voting :
            return ConfiguredPipeline<AAGrouping, Criteria, Voting>();
        default:
            throw std::runtime_error( "Undefined Strategy" );
    }
};

template<typename AAGrouping>
PipelineVariant getConfiguredPipeline( CriteriaEnum criteria, StrategyEnum strategy )
{
    switch (criteria)
    {
        case CriteriaEnum::ChiSquared :
            return getConfiguredPipeline<AAGrouping, ChiSquared>( strategy );
        case CriteriaEnum::Cosine :
            return getConfiguredPipeline<AAGrouping, Cosine>( strategy );
        case CriteriaEnum::KullbackLeiblerDiv:
            return getConfiguredPipeline<AAGrouping, KullbackLeiblerDivergence>( strategy );
        case CriteriaEnum::Gaussian :
            return getConfiguredPipeline<AAGrouping, Gaussian>( strategy );
        case CriteriaEnum::Intersection :
            return getConfiguredPipeline<AAGrouping, Intersection>( strategy );
        case CriteriaEnum::DensityPowerDivergence1 :
            return getConfiguredPipeline<AAGrouping, DensityPowerDivergence1>( strategy );
        case CriteriaEnum::DensityPowerDivergence2 :
            return getConfiguredPipeline<AAGrouping, DensityPowerDivergence2>( strategy );
        case CriteriaEnum::DensityPowerDivergence3:
            return getConfiguredPipeline<AAGrouping, DensityPowerDivergence3>( strategy );
        case CriteriaEnum::ItakuraSaitu :
            return getConfiguredPipeline<AAGrouping, ItakuraSaitu>( strategy );
        default:
            throw std::runtime_error( "Undefined Strategy" );
    }
};


PipelineVariant getConfiguredPipeline( AminoAcidGroupingEnum grouping, CriteriaEnum criteria,
                                       StrategyEnum strategy )
{
    switch (grouping)
    {
        case AminoAcidGroupingEnum::NoGrouping20:
            return getConfiguredPipeline<AAGrouping_NOGROUPING20>( criteria, strategy );
        case AminoAcidGroupingEnum::DIAMOND11 :
            return getConfiguredPipeline<AAGrouping_DIAMOND11>( criteria, strategy );
        case AminoAcidGroupingEnum::OLFER8 :
            return getConfiguredPipeline<AAGrouping_OLFER8>( criteria, strategy );
        case AminoAcidGroupingEnum::OLFER15 :
            return getConfiguredPipeline<AAGrouping_OLFER15>( criteria, strategy );
        default:
            throw std::runtime_error( "Undefined Strategy" );

    }
}

PipelineVariant getConfiguredPipeline( const std::string &groupingName,
                                       const std::string &criteria,
                                       const std::string &strategy )
{
    const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
    const CriteriaEnum criteriaLabel = CriteriaLabels.at( criteria );
    const StrategyEnum strategyLabel = ClassificationStrategyLabel.at( strategy );
    return getConfiguredPipeline( groupingLabel, criteriaLabel, strategyLabel );
}


#endif //MARKOVIAN_FEATURES_CONFIGUREDPIPELINE_HPP
