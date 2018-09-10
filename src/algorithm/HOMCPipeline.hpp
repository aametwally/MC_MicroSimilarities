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
#include "Histogram.hpp"
#include "HOMC.hpp"
#include "HOMCFeatures.hpp"

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


using MC::Selection;

template<typename Grouping, typename Criteria, typename Strategy>
class HOMCPipeline
{
public:

private:

    using PriorityQueue = typename MatchSet<Score>::Queue;



    using HOMCF = MC::HOMCFeatures<Grouping>;
    using HOMCP =  MC::HOMC<Grouping>;
    using HOMCOps = MC::Ops<Grouping>;

    using BackboneProfiles =  typename HOMCP::BackboneProfiles;
    using Histogram = typename HOMCP::Histogram;
    using DoubleSeries = typename HOMCP::ProbabilitisByOrder;
    using KernelsSeries = typename HOMCP::HistogramSeriesByOrder;

    using HeteroHistograms = typename HOMCP::HeteroHistograms;
    using HeteroHistogramsFeatures = typename HOMCP::HeteroHistogramsFeatures;

    static constexpr const char *LOADING = "loading";
    static constexpr const char *PREPROCESSING = "preprocessing";
    static constexpr const char *TRAINING = "training";
    static constexpr const char *CLASSIFICATION = "classification";

    using LeaderBoard = typename std::conditional<std::is_same<Voting, Strategy>::value, ClassificationCandidates<Score>, ClassificationCandidates<Criteria> >::type;

public:

    template< typename Entries >
    static std::vector<LabeledEntry>
    reducedAlphabetEntries( Entries &&entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( std::forward<Entries>( entries ));
    }

    static std::vector<LeaderBoard>
    classify_VALIDATION(
            const std::vector<std::string> &queries,
            const std::vector<std::string> &trueLabels,
            BackboneProfiles &&targets,
            std::map<std::string, std::vector<std::string >> &&trainingClusters,
            const Selection &selection,
            Order mnOrder, Order mxOrder )
    {
        assert( queries.size() == trueLabels.size());

        auto results = predict( queries, targets, std::move( trainingClusters ), selection, mnOrder, mxOrder );
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
            const BackboneProfiles &targets,
            std::map<std::string, std::vector<std::string >> &&kernelRelevance,
            const Selection &selection,
            Order mxOrder )
    {
        std::vector<PriorityQueue> results;
        for (auto &seq : queries)
        {
            std::map<std::string, double> accumulator;

            HOMCP query( mxOrder );
            query.train( {seq} );
//            query = MP::filter( std::move( query ), selection );
            for (const auto &[order, isoKernels] : query.histograms())
                for (const auto &[id, k1] : isoKernels.get())
                {
                    const auto p1 = query.probabilitisByOrder( mxOrder, id );
                    const auto s1 = query.kernelsByOrder( mxOrder, id );
                    PriorityQueue pq( targets.size());

                    for (const auto &[clusterName, profile] : targets)
                    {
                        auto s2 = profile.kernelsByOrder( mxOrder, id );


                        auto val = KernelsSeries::sum( s1, s2,
                                                       [=]( const Histogram &t1, const Histogram &t2 ) {
                                                           return Criteria::measure( t1, t2 );
                                                       } );
                        pq.emplace( clusterName, val );

                    }
                    pq.forTopK( 1, [&]( const auto &candidate ) {
                        std::string label = candidate.getLabel();
//                        FeatureSeries relevance( kernelRelevance, mxOrder, id );
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
            const BackboneProfiles &targets,
            std::map<std::string, std::vector<std::string >> &&trainingClusters,
            const Selection &selection,
            Order mnOrder,
            Order mxOrder )
    {
        const auto relevance =
                HOMCF::minMaxScale(
                        HOMCF::histogramRelevance_ALL2WITHIN_WEIGHTED( trainingClusters, mnOrder, mxOrder, selection ));
        std::vector<PriorityQueue> results;
        for (auto &seq : queries)
        {
            std::map<std::string, double> voter;

            if ( auto query = HOMCOps::filter( HOMCP( {seq}, mnOrder , mxOrder ), selection ); query )
                for (const auto &[order, isoKernels] : query->histograms())
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

            PriorityQueue vPQ( targets.size());
            for (const auto &[id, votes]: voter)
                vPQ.emplace( id, votes );
            results.emplace_back( std::move( vPQ ));
        }
        return results;
    }

    static std::vector<PriorityQueue>
    predict_ACCUMULATIVE( const std::vector<std::string> &queries,
                          const BackboneProfiles &targets,
                          std::map<std::string, std::vector<std::string >> &&trainingClusters,
                          const Selection &selection,
                          Order mnOrder,
                          Order mxOrder )
    {
        auto relevance =
                HOMCF::minMaxScaleByOrder( HOMCF::histogramRelevance_MAX2MIN_WEIGHTED( trainingClusters,
                                                                                 mnOrder , mxOrder , selection ));

        std::vector<PriorityQueue> results;
        for (auto &seq : queries)
        {
            PriorityQueue matchSet( targets.size());

            if ( auto queryOpt = HOMCOps::filter( HOMCP( {seq}, mnOrder, mxOrder ), selection ); queryOpt )
            {
                for (const auto &[clusterId, profile] : targets)
                {
                    double sum = 0;
                    for (const auto &[order, isoKernels] : queryOpt->histograms())
                        for (const auto &[id, kernel1] : isoKernels.get())
                        {
                            double score = getOr( relevance, order, id, double( 0 ));
                            auto k2 = profile.kernel( order, id );
                            if ( k2 )
                            {
                                sum += Criteria::measure( kernel1, k2.value().get()) + score;
                            }
                        }
                    matchSet.emplace( clusterId, sum );
                }
            }
            results.emplace_back( std::move( matchSet ));
        }

        return results;
    }

    static std::vector<PriorityQueue>
    predict( const std::vector<std::string> &queries,
             const BackboneProfiles &targets,
             std::map<std::string, std::vector<std::string >> &&trainingClusters,
             const Selection &selection,
             Order mnOrder, Order mxOrder )
    {
        if ( std::is_same<Strategy, Voting>::value )
            return predict_VOTING2( queries, targets, std::move( trainingClusters ), selection, mnOrder, mxOrder );
        else if ( std::is_same<Strategy, Accumulative>::value )
            return predict_ACCUMULATIVE( queries, targets, std::move( trainingClusters ), selection, mnOrder, mxOrder );
        else throw std::runtime_error( "Undefined Strategy!" );
    }

    static std::pair<Selection, BackboneProfiles>
    featureSelection( const std::map<std::string, std::vector<std::string> > &trainingClusters,
                      Order mnOrder, Order mxOrder )
    {
        auto selection = HOMCOps::withinJointAllUnionKernels( trainingClusters, mnOrder, mxOrder, 0.05 );
//        auto scoredFeatures = MF::histogramRelevance_ALL2WITHIN_UNIFORM( trainingClusters, order, selection );
//        auto scoredSelection = MF::filter( scoredFeatures, 0.75 );
        auto trainedProfiles = HOMCOps::train( std::move( trainingClusters ), mnOrder, mxOrder, selection );
        return std::make_pair( std::move( selection ), std::move( trainedProfiles ));
    }

    void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries,  Order mnOrder, Order mxOrder, size_t k )
    {
        std::set<std::string> labels;
        for (const auto &entry : entries)
            labels.insert( entry.getLabel());

        using Folds = std::vector<std::vector<std::pair<std::string, std::string >>>;

        auto groupedEntries = LabeledEntry::groupSequencesByLabels( reducedAlphabetEntries( std::move( entries )));
//            fmt::print( "[All Sequences:{}]\n", entries.size());
//            auto labels = keys( groupedEntries );
//            for (auto &l : labels) l = fmt::format( "{}({})", l, groupedEntries.at( l ).size());
//            fmt::print( "[Clusters:{}][{}]\n",
//                        groupedEntries.size(),
//                        io::join( labels, "|" ));
        const Folds folds = kFoldStratifiedSplit( std::move( groupedEntries ), k );

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
            auto trainedProfiles = HOMCOps::train( trainingClusters, mnOrder, mxOrder );
            auto[selection, filteredProfiles] = featureSelection( trainingClusters, mnOrder, mxOrder );
            auto classificationResults = classify_VALIDATION( test, tLabels, std::move( filteredProfiles ),
                                                              std::move( trainingClusters ),
                                                              selection, mnOrder, mxOrder );

            for (const auto &classification : classificationResults)
            {
                if( auto prediction = classification.bestMatch() ;prediction )
                {
                    ++histogram[classification.trueClusterRank()];
                    validation.countInstance( i, prediction.value(), classification.trueCluster());
                }
                else {
                    ++histogram[-1];
                    validation.countInstance( i, "unclassified", classification.trueCluster());
                }
            }
        }

        validation.printReport();

        fmt::print( "True Classification Histogram:\n" );

        for (auto &[k, v] : histogram)
        {
            if ( k == -1 )
                fmt::print( "[{}:{}]", "Unclassified", v );
            else
                fmt::print( "[{}:{}]", fmt::format( "Rank{}", k ), v );
        }
        fmt::print("\n");
    }

};

template<typename...>
struct StrategiesList
{
};
using SupportedStrategies = StrategiesList<Voting, Accumulative>;


using PipelineVariant = MakeVariantType<HOMCPipeline,
        SupportedAAGrouping,
        SupportedCriteria,
        SupportedStrategies>;

template<typename AAGrouping, typename Criteria>
PipelineVariant getConfiguredPipeline( StrategyEnum strategy )
{
    switch (strategy)
    {
        case StrategyEnum::Accumulative :
            return HOMCPipeline<AAGrouping, Criteria, Accumulative>();
        case StrategyEnum::Voting :
            return HOMCPipeline<AAGrouping, Criteria, Voting>();
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
        case CriteriaEnum::Bhattacharyya :
            return getConfiguredPipeline<AAGrouping, Bhattacharyya>( strategy );
        case CriteriaEnum::Hellinger :
            return getConfiguredPipeline<AAGrouping, Hellinger>( strategy );
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
