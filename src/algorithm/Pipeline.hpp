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

#include "SVMMCParameters.hpp"
#include "SVMConfusionMC.hpp"
#include "KNNConfusionMC.hpp"

#include "SimilarityMetrics.hpp"


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
            {"lsmc", MCModelsEnum::LocalitySensitiveMC}
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
        SVM_Propensity,
        KNN_Propensity,
        KMERS
    };

    static const std::map<std::string, ClassificationMethod> ClassificationMethodLabel = {
            {"voting",         ClassificationMethod::Voting},
            {"voting_bg",      ClassificationMethod::Voting_WBG},
            {"acc",            ClassificationMethod::Accumulative},
            {"acc_bg",         ClassificationMethod::Accumulative_WBG},
            {"propensity",     ClassificationMethod::Propensity},
            {"propensity_bg",  ClassificationMethod::Propensity_WBG},
            {"svm",            ClassificationMethod::SVM},
            {"svm_propensity", ClassificationMethod::SVM_Propensity},
            {"knn_propensity", ClassificationMethod::KNN_Propensity},
            {"kmers",          ClassificationMethod::KMERS}
    };

    template<typename Grouping>
    class Pipeline
    {

    private:
        using MCF = MCFeatures<Grouping>;

        using AbstractModel = AbstractMC<Grouping>;

        using BackboneProfiles =  typename AbstractModel::BackboneProfiles;
        using BackboneProfile =  typename AbstractModel::BackboneProfile;

        using Histogram = typename AbstractModel::Histogram;

        using HeteroHistograms = typename AbstractModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename AbstractModel::HeteroHistogramsFeatures;

        using Similarity = MetricFunction<Histogram>;

        static constexpr const char *LOADING = "loading";
        static constexpr const char *PREPROCESSING = "preprocessing";
        static constexpr const char *TRAINING = "training";
        static constexpr const char *CLASSIFICATION = "classification";

        using PriorityQueue = typename MatchSet<Score>::Queue<std::string_view>;
        using LeaderBoard = ClassificationCandidates<Score>;

    public:
        Pipeline( ModelGenerator<Grouping> modelTrainer, Similarity similarity )
                : _modelTrainer( modelTrainer ),
                  _similarity( similarity )
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
                const std::vector<std::string_view> &trueLabels,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection,
                const ClassificationMethod classificationStrategy ) const
        {
            assert( queries.size() == trueLabels.size());

            auto results = predict( queries, targets, trainingClusters, selection,
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
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            std::vector<PriorityQueue> results;
            for (const auto &seq : queries)
            {
                auto reversedSeq = reverse( seq );

                auto kmers = extractKmersWithCounts( seq, 3, 6 );
//                for (auto&[rkmer, count] : extractKmers( reversedSeq, 3, 8 ))
//                    kmers[rkmer] += count;

                const size_t kTop = kmers.size() / 1.5;
                std::map<std::string_view, PriorityQueue> propensity;


                for (auto&[label, backbone] :targets)
                {
                    using It = std::map<std::string_view, PriorityQueue>::iterator;
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
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            auto trainedHistograms = AbstractModel::trainIndividuals( trainingClusters, _modelTrainer, selection );
            auto[withinClassRadius, populationRadius] = MCF::populationRadius( trainedHistograms, selection );
            const auto relevance3 =
                    MCF::minMaxScale(
                            MCF::histogramRelevance_ALL2WITHIN_UNIFORM( withinClassRadius, populationRadius ));
            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                std::map<std::string_view, double> voter;

                if ( auto query = _modelTrainer( seq, selection ); *query )
                    for (const auto &[order, isoKernels] : query->histograms().get())
                    {
                        for (const auto &[id, k1] : isoKernels)
                        {
                            PriorityQueue pq( targets.size());
                            for (const auto &[clusterName, profile] : targets)
                            {
                                if ( auto k2Opt = profile->histogram( order, id ); k2Opt )
                                {
                                    auto val = _similarity( k1, k2Opt.value().get());
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
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            auto clustersIR = MCF::informationRadius_UNIFORM( targets, selection );
            auto backgroundIR = MCF::informationRadius_UNIFORM( background, selection );

            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                std::map<std::string_view, double> voter;

                if ( auto query = _modelTrainer( seq, selection ); *query )
                    for (const auto &[order, isoHistograms] : query->histograms().get())
                    {
                        for (const auto &[id, histogram1] : isoHistograms)
                        {
                            PriorityQueue pq( targets.size());
                            for (const auto &[clusterName, profile] : targets)
                            {
                                auto &bg = background.at( clusterName );
                                auto histogram2 = profile->histogram( order, id );
                                auto hBG = bg->histogram( order, id );
                                if ( histogram2 && hBG )
                                {
//                                    auto val = _similarity( histogram1, histogram2->get()) -
//                                               _similarity( histogram1, hBG->get());
                                    auto val = _similarity( histogram1 - hBG->get(), histogram2->get() - hBG->get());
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
                              const std::map<std::string, std::vector<std::string >> &trainingClusters,
                              const Selection &selection ) const
        {


            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                PriorityQueue matchSet( targets.size());

                if ( auto query = _modelTrainer( seq, selection ); *query )
                {
                    for (const auto &[clusterId, profile] : targets)
                    {
                        double sum = 0;
                        for (const auto &[order, isoKernels] : query->histograms().get())
                            for (const auto &[id, kernel1] : isoKernels)
                            {

                                auto k2 = profile->histogram( order, id );
                                if ( k2 )
                                {
                                    sum += _similarity( kernel1, k2.value().get());
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
                                  const std::map<std::string, std::vector<std::string >> &trainingClusters,
                                  const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            auto trainedHistograms = AbstractModel::trainIndividuals( trainingClusters, _modelTrainer, selection );
            auto[withinClassRadius, populationRadius] = MCF::populationRadius( trainedHistograms, selection );
            const auto relevance =
                    MCF::minMaxScale(
                            MCF::histogramRelevance_ALL2MIN_UNIFORM( withinClassRadius, populationRadius ));

            std::vector<PriorityQueue> results;
            for (auto &seq : queries)
            {
                PriorityQueue matchSet( targets.size());

                if ( auto query = _modelTrainer( seq, selection ); *query )
                {
                    for (const auto &[clusterId, profile] : targets)
                    {
                        auto &bg = background.at( clusterId );
                        double sum = 0;
                        for (const auto &[order, isoKernels] : query->histograms().get())
                            for (const auto &[id, histogram1] : isoKernels)
                            {
                                double score = getOr( relevance, order, id, double( 0 ));
                                auto histogram2 = profile->histogram( order, id );
                                auto hBG = bg->histogram( order, id );
                                if ( histogram2 && hBG )
                                {
//                                    sum += _similarity( histogram1, histogram2->get()) -
//                                           _similarity( histogram1, hBG->get());
                                    sum += _similarity( histogram1 - hBG->get(), histogram2->get() - hBG->get());
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
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
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
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

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
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            SVMMCParameters<Grouping> svm( _modelTrainer );
            svm.fit( targets, background, trainingClusters );
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
        predict_SVM_Propensity(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            SVMConfusionMC<Grouping> svm( _modelTrainer );
            svm.fit( targets, background, trainingClusters );
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
        predict_KNN_Propensity(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            KNNConfusionMC<Grouping> knn( 3 );
            knn.fit( targets, background, trainingClusters );
            auto predicted = knn.predict( queries );

            std::vector<PriorityQueue> rankedPredictions;

            for (auto &predictedClass : predicted)
            {
                auto it = targets.find( std::string( predictedClass ));
                if ( it == targets.end())
                    throw std::runtime_error( fmt::format( "Unexpected label {}", predictedClass ));

                PriorityQueue matchSet( 1 );
                matchSet.emplace( predictedClass, 0 );
                rankedPredictions.emplace_back( std::move( matchSet ));
            }

            return rankedPredictions;
        }

        BackboneProfiles
        backgroundProfiles( const std::map<std::string, std::vector<std::string >> &trainingSequences,
                            const Selection &selection ) const
        {
            BackboneProfiles background;
            for (auto &[label, _] : trainingSequences)
            {
                std::vector<std::string> backgroundSequences;
                for (auto&[bgLabel, bgSequences] : trainingSequences)
                {
                    if ( bgLabel == label ) continue;
                    for (auto &s : bgSequences)
                        backgroundSequences.push_back( s );
                }
                background.emplace( label, _modelTrainer( backgroundSequences, selection ));
            }
            return background;
        }


        BackboneProfile
        backgroundProfile( const std::map<std::string, std::vector<std::string >> &trainingSequences,
                           const Selection &selection ) const
        {
            BackboneProfiles background;
            std::vector<std::string_view> backgroundSequences;
            for (auto &[label, seqs] : trainingSequences)
                for (auto &s : seqs)
                    backgroundSequences.push_back( s );

            return _modelTrainer( backgroundSequences, selection );
        }

        std::vector<PriorityQueue>
        predict( const std::vector<std::string> &queries,
                 const BackboneProfiles &targets,
                 const std::map<std::string, std::vector<std::string >> &trainingClusters,
                 const Selection &selection,
                 const ClassificationMethod classificationStrategy ) const
        {
            switch (classificationStrategy)
            {
                case ClassificationMethod::Accumulative :
                    return predict_ACCUMULATIVE( queries, targets, trainingClusters, selection );
                case ClassificationMethod::Accumulative_WBG :
                    return predict_ACCUMULATIVE_WBG( queries, targets, trainingClusters, selection );
                case ClassificationMethod::Voting :
                    return predict_VOTING( queries, targets, trainingClusters, selection );
                case ClassificationMethod::Voting_WBG :
                    return predict_VOTING_WBG( queries, targets, trainingClusters, selection );
                case ClassificationMethod::Propensity :
                    return predict_PROPENSITY( queries, targets, trainingClusters, selection );
                case ClassificationMethod::Propensity_WBG :
                    return predict_PROPENSITY_WBG( queries, targets, trainingClusters, selection );
                case ClassificationMethod::SVM :
                    return predict_SVM( queries, targets, trainingClusters, selection );
                case ClassificationMethod::SVM_Propensity :
                    return predict_SVM_Propensity( queries, targets, trainingClusters, selection );
                case ClassificationMethod::KNN_Propensity :
                    return predict_KNN_Propensity( queries, targets, trainingClusters, selection );
                case ClassificationMethod::KMERS :
                    return predict_KMERS( queries, targets, trainingClusters, selection );
                default:
                    throw std::runtime_error( "Undefined Strategy" );
            }
        }

        std::pair<Selection, BackboneProfiles>
        featureSelection( const std::map<std::string, std::vector<std::string> > &trainingClusters )
        {
            auto selection = AbstractModel::withinJointAllUnionKernels( trainingClusters, _modelTrainer, 0.3 );
            auto trainedProfiles = AbstractModel::train( std::move( trainingClusters ), _modelTrainer, selection );

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
                std::vector<std::string> sequences;
                std::vector<std::string_view> labels;
                for (const auto &item : items)
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
                const auto[test, tLabels] = extractTest( folds.at( i ));
                const auto[selection, filteredProfiles] = featureSelection( trainingClusters );
                auto classificationResults = classify_VALIDATION( test, tLabels, filteredProfiles,
                                                                  trainingClusters, selection,
                                                                  classificationStrategy );

                for (const auto &classification : classificationResults)
                {
                    if ( auto prediction = classification.bestMatch();prediction )
                    {
                        auto it = trainingClusters.find( std::string( prediction.value()));
                        if ( it == trainingClusters.end())
                            throw std::runtime_error( fmt::format( "Unexpected label {}", prediction.value()));

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
        const ModelGenerator<Grouping> _modelTrainer;
        const Similarity _similarity;

    };


    using PipelineVariant = MakeVariantType<Pipeline, SupportedAAGrouping>;


    template<typename AAGrouping, typename Similarity>
    PipelineVariant getConfiguredPipeline( MCModelsEnum model, Order mnOrder, Order mxOrder, Similarity similarity )
    {
        using M = AbstractMC<AAGrouping>; // M: Model
        using MG = ModelGenerator<AAGrouping>;
        using RMC = MC<AAGrouping>;
        using ROMC = RangedOrderMC<AAGrouping>;
        using ZMC = ZYMC<AAGrouping>;
        using LSMCM = LSMC<AAGrouping>;


        switch (model)
        {
            case MCModelsEnum::RegularMC :
                return Pipeline<AAGrouping>( MG::template create<RMC>( mxOrder ), similarity );
            case MCModelsEnum::RangedOrderMC :
                return Pipeline<AAGrouping>( MG::template create<ROMC>( mnOrder, mxOrder ), similarity );
            case MCModelsEnum::ZhengYuanMC :
                return Pipeline<AAGrouping>( MG::template create<ZMC>( mxOrder ), similarity );
            case MCModelsEnum::LocalitySensitiveMC :
                return Pipeline<AAGrouping>( MG::template create<LSMCM>( mxOrder ), similarity );
            default:
                throw std::runtime_error( "Undefined Strategy" );
        }
    };


    template<typename AAGrouping>
    PipelineVariant getConfiguredPipeline( CriteriaEnum criteria,
                                           MCModelsEnum model, Order mnOrder, Order mxOrder )
    {
        using AbstractModel = AbstractMC<AAGrouping>;
        using Histogram = typename AbstractModel::Histogram;

        switch (criteria)
        {
            case CriteriaEnum::ChiSquared :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, ChiSquared::function<Histogram> );
            case CriteriaEnum::Cosine :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, Cosine::function<Histogram> );
            case CriteriaEnum::Dot :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, Dot::function<Histogram> );
            case CriteriaEnum::KullbackLeiblerDiv:
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, KullbackLeiblerDivergence::function<Histogram> );
            case CriteriaEnum::Gaussian :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, Gaussian::function<Histogram> );
            case CriteriaEnum::Intersection :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, Intersection::function<Histogram> );
            case CriteriaEnum::DensityPowerDivergence1 :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, DensityPowerDivergence1::function<Histogram> );
            case CriteriaEnum::DensityPowerDivergence2 :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, DensityPowerDivergence2::function<Histogram> );
            case CriteriaEnum::DensityPowerDivergence3:
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, DensityPowerDivergence3::function<Histogram> );
            case CriteriaEnum::ItakuraSaitu :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, ItakuraSaitu::function<Histogram> );
            case CriteriaEnum::Bhattacharyya :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, Bhattacharyya::function<Histogram> );
            case CriteriaEnum::Hellinger :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, Hellinger::function<Histogram> );
            case CriteriaEnum::MaxIntersection :
                return getConfiguredPipeline<AAGrouping>(
                        model, mnOrder, mxOrder, MaxIntersection::function<Histogram> );
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
