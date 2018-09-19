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
#include "EnsembleCrossValidation.hpp"
#include "Histogram.hpp"

#include "AbstractMC.hpp"
#include "MC.hpp"
#include "RangedOrderMC.hpp"
#include "ZYMC.hpp"
#include "LSMC.hpp"
#include "MCFeatures.hpp"

#include "MCPropensityClassifier.hpp"
#include "MCKmersClassifier.hpp"
#include "MicroSimilarityVotingClassifier.hpp"
#include "MacroSimilarityClassifier.hpp"
#include "SVMMCParameters.hpp"
#include "KNNMCParameters.hpp"
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

        std::vector<std::string_view>
        predict( const std::vector<std::string> &queries,
                 const BackboneProfiles &targets,
                 const BackboneProfiles &background,
                 const std::map<std::string_view, std::vector<std::string >> &trainingClusters,
                 const Selection &selection,
                 const ClassificationEnum classificationStrategy ) const
        {
            switch (classificationStrategy)
            {
                case ClassificationEnum::Accumulative :
                {
                    auto model = MacroSimilarityClassifier<Grouping>(
                            targets, background, selection, _modelTrainer, _similarity );
                    return model.predict( queries );
                }
                case ClassificationEnum::Voting :
                {
                    auto model = MicroSimilarityVotingClassifier<Grouping>(
                            targets, background, selection, _modelTrainer, _similarity );
                    return model.predict( queries );
                }
                case ClassificationEnum::Propensity :
                {
                    MCPropensityClassifier<Grouping> classifier( targets, background );
                    return classifier.predict( queries );
                }
                case ClassificationEnum::SVM :
                {
                    SVMMCParameters<Grouping> svm( _modelTrainer );
                    svm.fit( targets, background, trainingClusters );
                    return svm.predict( queries );
                }
                case ClassificationEnum::KNN :
                {
                    KNNMCParameters<Grouping> knn( _modelTrainer, _similarity );
                    knn.fit( targets, background, trainingClusters );
                    return knn.predict( queries );

                }
                case ClassificationEnum::SVM_Stack :
                {
                    SVMConfusionMC<Grouping> svm( _modelTrainer );
                    svm.fit( targets, background, trainingClusters, _modelTrainer, selection, _similarity );
                    return svm.predict( queries );
                }
                case ClassificationEnum::KNN_Stack :
                {
                    KNNConfusionMC<Grouping> knn( 3 );
                    knn.fit( targets, background, trainingClusters, _modelTrainer, selection, _similarity );
                    return knn.predict( queries );
                }
                case ClassificationEnum::KMERS :
                {
                    MCKmersClassifier<Grouping> classifier( targets, background );
                    return classifier.predict( queries );
                }
                default:
                    throw std::runtime_error( "Undefined Strategy" );
            }
        }

        std::pair<Selection, BackboneProfiles>
        featureSelection( const std::map<std::string_view, std::vector<std::string> > &trainingClusters )
        {
            auto selection = AbstractModel::withinJointAllUnionKernels( trainingClusters, _modelTrainer, 0.3 );
            auto trainedProfiles = AbstractModel::train( trainingClusters, _modelTrainer, selection );

            return std::make_pair( std::move( selection ), std::move( trainedProfiles ));
        }

        void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries, const size_t k,
                                     const std::vector<std::string> &classificationStrategy )
        {

            std::set<std::string> classifiers;
            for (const auto &classifier : classificationStrategy)
                classifiers.insert( classifier );

            std::set<std::string> labels;
            for (const auto &entry : entries)
                labels.insert( entry.getLabel());
            auto viewLabels = std::set<std::string_view>( labels.cbegin(), labels.cend());

            using Fold = std::vector<std::pair<std::string, LabeledEntry >>;
            using FoldSequences = std::vector<std::pair<std::string, std::string >>;
            using Folds = std::vector<Fold>;
            using FoldsSequences = std::vector<FoldSequences>;

            auto extractSequences = []( const Folds &folds ) {
                FoldsSequences fSequences;
                std::transform( folds.cbegin(), folds.cend(),
                                std::back_inserter( fSequences ), []( const Fold &f ) {
                            FoldSequences foldSequences;
                            std::transform( f.cbegin(), f.cend(), std::back_inserter( foldSequences ),
                                            []( const auto &p ) {
                                                return std::make_pair( p.first, p.second.getSequence());
                                            } );
                            return foldSequences;
                        } );
                return fSequences;
            };

            auto groupedEntries = LabeledEntry::groupEntriesByLabels( reducedAlphabetEntries( std::move( entries )));
//            fmt::print( "[All Sequences:{}]\n", entries.size());
//            auto labels = keys( groupedEntries );
//            for (auto &l : labels) l = fmt::format( "{}({})", l, groupedEntries.at( l ).size());
//            fmt::print( "[Clusters:{}][{}]\n",
//                        groupedEntries.size(),
//                        io::join( labels, "|" ));
            const Folds folds = kFoldStratifiedSplit( std::move( groupedEntries ), k );
            const FoldsSequences sFolds = extractSequences( folds );

            auto unzip = []( const std::vector<std::pair<std::string, LabeledEntry >> &items ) {
                std::vector<std::string_view> ids;
                std::vector<std::string> sequences;
                std::vector<std::string_view> ls;
                for (const auto &item : items)
                {
                    ls.push_back( item.first );
                    sequences.push_back( item.second.getSequence());
                    ids.push_back( item.second.getMemberId());
                }
                return std::make_tuple( ids, sequences, ls );
            };

            std::map<std::string, CrossValidationStatistics<std::string_view >> validation;
            EnsembleCrossValidation<std::string_view> ensembleValidation( folds );

            for (auto &classifier : classifiers)
                validation[classifier] = CrossValidationStatistics( k, viewLabels );


            for (size_t i = 0; i < k; ++i)
            {
                auto trainingClusters = joinFoldsExceptK( sFolds, i );
                const auto[ids, queries, qLabels] = unzip( folds.at( i ));
                const auto[selection, trained] = featureSelection( trainingClusters );
                const BackboneProfiles background = AbstractModel::backgroundProfiles(
                        trainingClusters, _modelTrainer, selection );
                for (auto &classifier : classifiers)
                {
                    auto classifierEnum = ClassifierEnum.at( classifier );
                    auto predictions = predict( queries, trained, background,
                                                trainingClusters, selection,
                                                classifierEnum );

                    assert( predictions.size() == qLabels.size() && qLabels.size() == queries.size() &&
                            queries.size() == ids.size());

                    auto &cValidation = validation[classifier];
                    for (size_t proteinIdx = 0; proteinIdx < queries.size(); ++proteinIdx)
                    {
                        const auto &fold = folds.at( i );

                        const auto &id = fold.at( proteinIdx ).second.getMemberId();
                        const auto &label = fold.at( proteinIdx ).first;
                        const auto &prediction = predictions.at( proteinIdx );

                        cValidation.countInstance( i, prediction, label );
                        ensembleValidation.countInstance( i, classifier, id, prediction );
                    }
                }
            }

            for (auto &classifier : classifiers)
            {
                validation[classifier].printReport( classifier );
                fmt::print( "\n" );
            }

            for (auto&[ensemble, cv] : ensembleValidation.majorityVotingOverallAccuracy())
            {
                fmt::print( "Ensemble{{{}}} Cross-validation\n", io::join( ensemble, "," ));
                cv.printReport();
            }
        }

    private:
        const ModelGenerator<Grouping> _modelTrainer;
        const Similarity _similarity;
    };

    using PipelineVariant = MakeVariantType<Pipeline, SupportedAAGrouping>;

    template<typename AAGrouping, typename Similarity>
    PipelineVariant getConfiguredPipeline( MCModelsEnum model, Order mnOrder, Order mxOrder, Similarity similarity )
    {
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
