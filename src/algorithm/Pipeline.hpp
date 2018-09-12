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

#include "AbstractMC.hpp"
#include "MC.hpp"
#include "RangedOrderMC.hpp"
#include "ZYMC.hpp"
#include "LSMC.hpp"
#include "MCFeatures.hpp"
#include "MCOperations.hpp"

#include "SVMMarkovianModel.hpp"

#include "similarities.hpp"


namespace MC {

    enum class MCModelsEnum
    {
        RegularMC,
        RangedOrderMC,
        ZhengYuanMC,
        LocalitySensitiveMC
    };

    const std::map<std::string, MCModelsEnum> MCModelLabels{
            {"rmc",  MCModelsEnum::RegularMC},
            {"romc", MCModelsEnum::RangedOrderMC},
            {"zymc", MCModelsEnum::ZhengYuanMC},
            {"lsmc", MCModelsEnum ::LocalitySensitiveMC }
    };

    enum class ClassificationMethod
    {
        Voting,
        Voting_WBG,
        Accumulative,
        Accumulative_WBG,
        Propensity,
        Propensity_WBG,
        SVM,
        KMERS
    };

    static const std::map<std::string, ClassificationMethod> ClassificationMethodLabel = {
            {"voting",        ClassificationMethod::Voting},
            {"voting_bg",     ClassificationMethod::Voting_WBG},
            {"acc",           ClassificationMethod::Accumulative},
            {"acc_bg",        ClassificationMethod::Accumulative_WBG},
            {"propensity",    ClassificationMethod::Propensity},
            {"propensity_bg", ClassificationMethod::Propensity_WBG},
            {"svm",           ClassificationMethod::SVM},
            {"kmers",         ClassificationMethod::KMERS}
    };

    template<typename Grouping, typename Criteria>
    class Pipeline
    {

    private:
        using MCF = MCFeatures<Grouping>;
        using Ops = MCOps<Grouping>;

        using AbstractModel = AbstractMC<Grouping>;


        using ModelTrainer = typename AbstractModel::ModelTrainer;
        using HistogramsTrainer = typename AbstractModel::HistogramsTrainer;
        using BackboneProfiles =  typename AbstractModel::BackboneProfiles;

        using Histogram = typename AbstractModel::Histogram;

        using HeteroHistograms = typename AbstractModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename AbstractModel::HeteroHistogramsFeatures;

        static constexpr const char *LOADING = "loading";
        static constexpr const char *PREPROCESSING = "preprocessing";
        static constexpr const char *TRAINING = "training";
        static constexpr const char *CLASSIFICATION = "classification";

        using PriorityQueue = typename MatchSet<Score>::Queue;
        using LeaderBoard = ClassificationCandidates<Score>;

    public:
        Pipeline( ModelTrainer &&modelTrainer, HistogramsTrainer &&histogramsTrainer )
                : _modelTrainer( modelTrainer ), _histogramsTrainer( histogramsTrainer )
        {

        }

    public:

        template<typename Entries>
        static std::vector<LabeledEntry>
        reducedAlphabetEntries( Entries &&entries )
        {
            return LabeledEntry::reducedAlphabetEntries<Grouping>( std::forward<Entries>( entries ));
        }

        std::vector<LeaderBoard>
        classify_VALIDATION(
                const std::vector<std::string> &queries,
                const std::vector<std::string> &trueLabels,
                BackboneProfiles &&targets,
                std::map<std::string, std::vector<std::string >> &&trainingClusters,
                const Selection &selection,
                const ClassificationMethod classificationStrategy )
        {
            assert( queries.size() == trueLabels.size());

            auto results = predict( queries, targets, std::move( trainingClusters ), selection,
                                    classificationStrategy );
            assert( results.size() == queries.size());
            std::vector<LeaderBoard> classifications;
            for (auto i = 0; i < queries.size(); ++i)
            {
                classifications.emplace_back( trueLabels.at( i ),
                                              results.at( i ));
            }
            return classifications;
        }


        std::vector<PriorityQueue>
        predict_KMERS(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                std::map<std::string, std::vector<std::string >> &&trainingClusters,
                const Selection &selection )
        {
            BackboneProfiles background;
            for (auto &[label, _] : trainingClusters)
            {
                std::vector<std::string> backgroundSequences;
                for (auto&[bgLabel, bgSequences] : trainingClusters)
                {
                    if ( bgLabel == label ) continue;
                    for (auto &s : bgSequences)
                        backgroundSequences.push_back( s );
                }
                background.emplace( label, _modelTrainer( backgroundSequences, selection ));
            }

            std::vector<PriorityQueue> results;
            for (const auto &seq : queries)
            {
                auto reversedSeq = reverse( seq );

                auto kmers = extractKmersWithCounts( seq, 3, 6 );
//                for (auto&[rkmer, count] : extractKmers( reversedSeq, 3, 8 ))
//                    kmers[rkmer] += count;

                const size_t kTop = kmers.size() / 1.5;
                std::map<std::string, PriorityQueue> propensity;


                for (auto&[label, backbone] :targets)
                {
                    using It = std::map<std::string, PriorityQueue>::iterator;
                    It propensityIt;
                    std::tie( propensityIt, std::ignore ) = propensity.emplace( label, kmers.size());
                    PriorityQueue &_propensity = propensityIt->second;
                    for (auto &[kmer, count] : kmers)
                    {
                        auto &bg = background.at( label );
                        double logOdd = backbone->propensity( kmer ) - bg->propensity( kmer );
                        _propensity.emplace( kmer, logOdd * count );
                    }
                }


                PriorityQueue vPQ( targets.size());
                for (const auto &[label, propensities]:  propensity)
                {
                    double sum = 0;
                    propensities.forTopK( kTop, [&]( const auto &candidate, size_t index ) {
                        sum += candidate.getValue();
                    } );
                    vPQ.emplace( label, sum );
                }
                results.emplace_back( std::move( vPQ ));
            }
            return results;
        }

        std::vector<PriorityQueue>
        predict_VOTING(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                std::map<std::string, std::vector<std::string >> &&trainingClusters,
                const Selection &selection )
        {
            auto trainedHistograms = Ops::trainIndividuals( trainingClusters, _histogramsTrainer, selection );
            auto[withinClassRadius, populationRadius] = MCF::populationRadius( trainedHistograms, selection );
            const auto relevance3 =
                    MCF::minMaxScale(
                            MCF::histogramRelevance_ALL2WITHIN_UNIFORM( withinClassRadius, populationRadius ));
            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                std::map<std::string_view, double> voter;

                if ( auto query = _histogramsTrainer( {seq}, selection ); query )
                    for (const auto &[order, isoKernels] : query.value())
                    {
                        for (const auto &[id, k1] : isoKernels)
                        {
                            PriorityQueue pq( targets.size());
                            for (const auto &[clusterName, profile] : targets)
                            {
                                if ( auto k2Opt = profile->histogram( order, id ); k2Opt )
                                {
                                    auto val = Criteria::measure( k1, k2Opt.value().get());
                                    pq.emplace( clusterName, val );
                                }
                            }
                            double score = //getOr( populationRadius, order, id, double( 0 )) +
                                    //getOr( relevance1, order, id, double( 0 )) +
                                    //getOr( relevance2, order, id, double( 0 )) +
                                    getOr( relevance3, order, id, double( 0 ));
                            pq.forTopK( 5, [&]( const auto &candidate, size_t index ) {
                                std::string_view label = candidate.getLabel();
                                voter[label] += (score + 1) / (index + 1);
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


        std::vector<PriorityQueue>
        predict_VOTING_WBG(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                std::map<std::string, std::vector<std::string >> &&trainingClusters,
                const Selection &selection )
        {
            BackboneProfiles background;
            for (auto &[label, _] : trainingClusters)
            {
                std::vector<std::string> backgroundSequences;
                for (auto&[bgLabel, bgSequences] : trainingClusters)
                {
                    if ( bgLabel == label ) continue;
                    for (auto &s : bgSequences)
                        backgroundSequences.push_back( s );
                }
                background.emplace( label, _modelTrainer( backgroundSequences, selection ));
            }

            auto clustersIR = MCF::informationRadius_UNIFORM( targets, selection );
            auto backgroundIR = MCF::informationRadius_UNIFORM( background, selection );

            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                std::map<std::string_view, double> voter;

                if ( auto query = _histogramsTrainer( {seq}, selection ); query )
                    for (const auto &[order, isoHistograms] : query.value())
                    {
                        for (const auto &[id, histogram1] : isoHistograms)
                        {
                            PriorityQueue pq( targets.size());
                            for (const auto &[clusterName, profile] : targets)
                            {
                                auto &bg = background.at( clusterName );
                                auto histogram2 = profile->histogram( order, id );
                                auto bgHistogram = bg->histogram( order, id );
                                if ( histogram2 && bgHistogram )
                                {
                                    auto val = Criteria::measure( histogram1, histogram2->get()) -
                                               Criteria::measure( histogram1, bgHistogram->get());
                                    pq.emplace( clusterName, val );
                                }
                            }
                            double score = getOr( backgroundIR, order, id, double( 0 )) -
                                           getOr( clustersIR, order, id, double( 0 ));

                            pq.forTopK( 5, [&]( const auto &candidate, size_t index ) {
                                std::string_view label = candidate.getLabel();
                                voter[label] += (score + 1) / (index + 1);
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

        std::vector<PriorityQueue>
        predict_ACCUMULATIVE( const std::vector<std::string> &queries,
                              const BackboneProfiles &targets,
                              std::map<std::string, std::vector<std::string >> &&trainingClusters,
                              const Selection &selection )
        {


            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                PriorityQueue matchSet( targets.size());

                if ( auto queryOpt = _histogramsTrainer( {seq}, selection ); queryOpt )
                {
                    for (const auto &[clusterId, profile] : targets)
                    {
                        double sum = 0;
                        for (const auto &[order, isoKernels] : queryOpt.value())
                            for (const auto &[id, kernel1] : isoKernels)
                            {

                                auto k2 = profile->histogram( order, id );
                                if ( k2 )
                                {
                                    sum += Criteria::measure( kernel1, k2.value().get());
                                }
                            }
                        matchSet.emplace( clusterId, sum );
                    }
                }
                results.emplace_back( std::move( matchSet ));
            }

            return results;
        }

        std::vector<PriorityQueue>
        predict_ACCUMULATIVE_WBG( const std::vector<std::string> &queries,
                                  const BackboneProfiles &targets,
                                  std::map<std::string, std::vector<std::string >> &&trainingClusters,
                                  const Selection &selection )
        {
            BackboneProfiles background;
            for (auto &[label, _] : trainingClusters)
            {
                std::vector<std::string> backgroundSequences;
                for (auto&[bgLabel, bgSequences] : trainingClusters)
                {
                    if ( bgLabel == label ) continue;
                    for (auto &s : bgSequences)
                        backgroundSequences.push_back( s );
                }
                background.emplace( label, _modelTrainer( backgroundSequences, selection ));
            }

            auto trainedHistograms = Ops::trainIndividuals( trainingClusters, _histogramsTrainer, selection );
            auto[withinClassRadius, populationRadius] = MCF::populationRadius( trainedHistograms, selection );
            const auto relevance =
                    MCF::minMaxScale(
                            MCF::histogramRelevance_ALL2MIN_UNIFORM( withinClassRadius, populationRadius ));

            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                PriorityQueue matchSet( targets.size());

                if ( auto queryOpt = _histogramsTrainer( {seq}, selection ); queryOpt )
                {
                    for (const auto &[clusterId, profile] : targets)
                    {
                        auto &bg = background.at( clusterId );
                        double sum = 0;
                        for (const auto &[order, isoKernels] : queryOpt.value())
                            for (const auto &[id, histogram1] : isoKernels)
                            {
                                double score = getOr( relevance, order, id, double( 0 ));
                                auto histogram2 = profile->histogram( order, id );
                                auto hBG = bg->histogram( order, id );
                                if ( histogram2 && hBG )
                                {
                                    sum += Criteria::measure( histogram1, histogram2->get()) -
                                           Criteria::measure( histogram1, hBG->get());
                                }
                            }
                        matchSet.emplace( clusterId, sum );
                    }
                }
                results.emplace_back( std::move( matchSet ));
            }

            return results;
        }


        std::vector<PriorityQueue>
        predict_PROPENSITY(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                std::map<std::string, std::vector<std::string >> &&trainingClusters,
                const Selection &selection )
        {
            std::vector<PriorityQueue> rankedPredictions;
            for (auto &query : queries)
            {
                PriorityQueue matchSet( targets.size());
                for (auto&[label, backbone] :targets)
                {
                    matchSet.emplace( label, backbone->propensity( query ));
                }
                rankedPredictions.emplace_back( std::move( matchSet ));
            }
            return rankedPredictions;
        }

        std::vector<PriorityQueue>
        predict_PROPENSITY_WBG(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                std::map<std::string, std::vector<std::string >> &&trainingClusters,
                const Selection &selection )
        {
            BackboneProfiles background;
            for (auto &[label, _] : trainingClusters)
            {
                std::vector<std::string> backgroundSequences;
                for (auto&[bgLabel, bgSequences] : trainingClusters)
                {
                    if ( bgLabel == label ) continue;
                    for (auto &s : bgSequences)
                        backgroundSequences.push_back( s );
                }
                background.emplace( label, _modelTrainer( backgroundSequences, selection ));
            }

            std::vector<PriorityQueue> rankedPredictions;
            for (auto &query : queries)
            {
                PriorityQueue matchSet( targets.size());
                for (auto&[label, backbone] :targets)
                {
                    auto &bg = background.at( label );
                    double logOdd = backbone->propensity( query ) - bg->propensity( query );
                    matchSet.emplace( label, logOdd );
                }
                rankedPredictions.emplace_back( std::move( matchSet ));
            }
            return rankedPredictions;
        }

        std::vector<PriorityQueue>
        predict_SVM(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                std::map<std::string, std::vector<std::string >> &&trainingClusters,
                const Selection &selection )
        {
            SVMMarkovianModel<Grouping> svm( _histogramsTrainer );
            svm.fit( trainingClusters );
            auto predicted = svm.predict( queries );

            std::vector<PriorityQueue> rankedPredictions;

            for (auto &predictedClass : predicted)
            {
                PriorityQueue matchSet( 1 );
                matchSet.emplace( predictedClass, 0 );
                rankedPredictions.emplace_back( std::move( matchSet ));
            }

            return rankedPredictions;
        }


        std::vector<PriorityQueue>
        predict( const std::vector<std::string> &queries,
                 const BackboneProfiles &targets,
                 std::map<std::string, std::vector<std::string >> &&trainingClusters,
                 const Selection &selection,
                 const ClassificationMethod classificationStrategy )
        {
            switch (classificationStrategy)
            {
                case ClassificationMethod::Accumulative :
                    return predict_ACCUMULATIVE( queries, targets, std::move( trainingClusters ), selection );
                case ClassificationMethod::Accumulative_WBG :
                    return predict_ACCUMULATIVE_WBG( queries, targets, std::move( trainingClusters ), selection );
                case ClassificationMethod::Voting :
                    return predict_VOTING( queries, targets, std::move( trainingClusters ), selection );
                case ClassificationMethod::Voting_WBG :
                    return predict_VOTING_WBG( queries, targets, std::move( trainingClusters ), selection );
                case ClassificationMethod::Propensity :
                    return predict_PROPENSITY( queries, targets, std::move( trainingClusters ), selection );
                case ClassificationMethod::Propensity_WBG :
                    return predict_PROPENSITY_WBG( queries, targets, std::move( trainingClusters ), selection );
                case ClassificationMethod::SVM :
                    return predict_SVM( queries, targets, std::move( trainingClusters ), selection );
                case ClassificationMethod::KMERS :
                    return predict_KMERS( queries, targets, std::move( trainingClusters ), selection );
                default:
                    throw std::runtime_error( "Undefined Strategy" );
            }
        }

        std::pair<Selection, BackboneProfiles>
        featureSelection( const std::map<std::string, std::vector<std::string> > &trainingClusters )
        {
            auto selection = Ops::withinJointAllUnionKernels( trainingClusters, _histogramsTrainer, 0.3 );
            auto trainedProfiles = Ops::train( std::move( trainingClusters ), _modelTrainer, selection );

            return std::make_pair( std::move( selection ), std::move( trainedProfiles ));
        }

        void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries, size_t k,
                                     const ClassificationMethod classificationStrategy )
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
                auto[selection, filteredProfiles] = featureSelection( trainingClusters );
                auto classificationResults = classify_VALIDATION( test, tLabels, std::move( filteredProfiles ),
                                                                  std::move( trainingClusters ), selection,
                                                                  classificationStrategy );

                for (const auto &classification : classificationResults)
                {
                    if ( auto prediction = classification.bestMatch();prediction )
                    {
                        ++histogram[classification.trueClusterRank()];
                        validation.countInstance( i, prediction.value(), classification.trueCluster());
                    } else
                    {
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
            fmt::print( "\n" );
        }

    private:
        const ModelTrainer _modelTrainer;
        const HistogramsTrainer _histogramsTrainer;

    };


    using PipelineVariant = MakeVariantType<Pipeline,
            SupportedAAGrouping,
            SupportedCriteria>;


    template<typename AAGrouping, typename Criteria>
    PipelineVariant getConfiguredPipeline( MCModelsEnum model, Order mnOrder, Order mxOrder )
    {
        using RMC = MC<AAGrouping>;
        using ROMC = RangedOrderMC<AAGrouping>;
        using ZMC = ZYMC<AAGrouping>;
        using LSMCM = LSMC<AAGrouping>;

        switch (model)
        {
            case MCModelsEnum::RegularMC :
                return Pipeline<AAGrouping, Criteria>( RMC::getModelTrainer( mxOrder ),
                                                       RMC::getHistogramsTrainer( mxOrder ));
            case MCModelsEnum::RangedOrderMC :
                return Pipeline<AAGrouping, Criteria>( ROMC::getModelTrainer( mnOrder, mxOrder ),
                                                       ROMC::getHistogramsTrainer( mnOrder, mxOrder ));
            case MCModelsEnum::ZhengYuanMC :
                return Pipeline<AAGrouping, Criteria>( ZMC::getModelTrainer( mxOrder ),
                                                       ZMC::getHistogramsTrainer( mxOrder ));
            case MCModelsEnum::LocalitySensitiveMC :
                return Pipeline<AAGrouping, Criteria>( LSMCM::getLSMCTrainer( mxOrder ),
                                                       LSMCM::getLSMCHistogramsTrainer( mxOrder ));
            default:
                throw std::runtime_error( "Undefined Strategy" );
        }
    };


    template<typename AAGrouping>
    PipelineVariant getConfiguredPipeline( CriteriaEnum criteria,
                                           MCModelsEnum model, Order mnOrder, Order mxOrder )
    {
        switch (criteria)
        {
            case CriteriaEnum::ChiSquared :
                return getConfiguredPipeline<AAGrouping, ChiSquared>( model, mnOrder, mxOrder );
            case CriteriaEnum::Cosine :
                return getConfiguredPipeline<AAGrouping, Cosine>( model, mnOrder, mxOrder );
            case CriteriaEnum::KullbackLeiblerDiv:
                return getConfiguredPipeline<AAGrouping, KullbackLeiblerDivergence>( model, mnOrder,
                                                                                     mxOrder );
            case CriteriaEnum::Gaussian :
                return getConfiguredPipeline<AAGrouping, Gaussian>( model, mnOrder, mxOrder );
            case CriteriaEnum::Intersection :
                return getConfiguredPipeline<AAGrouping, Intersection>( model, mnOrder, mxOrder );
            case CriteriaEnum::DensityPowerDivergence1 :
                return getConfiguredPipeline<AAGrouping, DensityPowerDivergence1>( model, mnOrder, mxOrder );
            case CriteriaEnum::DensityPowerDivergence2 :
                return getConfiguredPipeline<AAGrouping, DensityPowerDivergence2>( model, mnOrder, mxOrder );
            case CriteriaEnum::DensityPowerDivergence3:
                return getConfiguredPipeline<AAGrouping, DensityPowerDivergence3>( model, mnOrder, mxOrder );
            case CriteriaEnum::ItakuraSaitu :
                return getConfiguredPipeline<AAGrouping, ItakuraSaitu>( model, mnOrder, mxOrder );
            case CriteriaEnum::Bhattacharyya :
                return getConfiguredPipeline<AAGrouping, Bhattacharyya>( model, mnOrder, mxOrder );
            case CriteriaEnum::Hellinger :
                return getConfiguredPipeline<AAGrouping, Hellinger>( model, mnOrder, mxOrder );
            default:
                throw std::runtime_error( "Undefined Strategy" );
        }
    };


    PipelineVariant getConfiguredPipeline( AminoAcidGroupingEnum grouping, CriteriaEnum criteria, MCModelsEnum model,
                                           Order mnOrder, Order mxOrder )
    {
        switch (grouping)
        {
            case AminoAcidGroupingEnum::NoGrouping20:
                return getConfiguredPipeline<AAGrouping_NOGROUPING20>( criteria, model, mnOrder, mxOrder );
//            case AminoAcidGroupingEnum::DIAMOND11 :
//                return getConfiguredPipeline<AAGrouping_DIAMOND11>( criteria, model, mnOrder, mxOrder );
//            case AminoAcidGroupingEnum::OFER8 :
//                return getConfiguredPipeline<AAGrouping_OFER8>( criteria, model, mnOrder, mxOrder );
            case AminoAcidGroupingEnum::OFER15 :
                return getConfiguredPipeline<AAGrouping_OFER15>( criteria, model, mnOrder, mxOrder );
            default:
                throw std::runtime_error( "Undefined Grouping" );

        }
    }

    PipelineVariant getConfiguredPipeline( const std::string &groupingName,
                                           const std::string &criteria,
                                           const std::string &model,
                                           Order mnOrder, Order mxOrder )
    {
        const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
        const CriteriaEnum criteriaLabel = CriteriaLabels.at( criteria );
        const MCModelsEnum modelLabel = MCModelLabels.at( model );

        return getConfiguredPipeline( groupingLabel, criteriaLabel, modelLabel, mnOrder, mxOrder );
    }

}
#endif //MARKOVIAN_FEATURES_CONFIGUREDPIPELINE_HPP
