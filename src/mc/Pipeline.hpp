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

#include "MCModels.hpp"
#include "MCFeatures.hpp"

#include "MCSegmentationClassifier.hpp"
#include "MCKmersClassifier.hpp"
#include "MCBasedMLModel.hpp"
#include "AbstractMCClassifier.hpp"
#include "MCDiscretizedScalesClassifier.hpp"

#include "AAIndexClustering.hpp"
#include "SimilarityMetrics.hpp"
#include "MCSegmentationClassifier.hpp"

namespace MC {
enum class MCModelsEnum
{
    RegularMC,
    ZhengYuanMC,
    GappedMC,
    RegularizedBinaryMC,
    RegularizedVectorsMC
};

const std::map<std::string, MCModelsEnum> MCModelLabels{
        {"mc",   MCModelsEnum::RegularMC},
        {"zymc", MCModelsEnum::ZhengYuanMC},
        {"gmc",  MCModelsEnum::GappedMC},
        {"bmc",  MCModelsEnum::RegularizedBinaryMC},
        {"vmc",  MCModelsEnum::RegularizedVectorsMC}
};

template<typename Grouping>
class Pipeline
{
private:
    static constexpr auto States = Grouping::StatesN;
    using MCF = MCFeatures<States>;

    using AbstractModel = AbstractMC<States>;

    using BackboneProfiles =  typename AbstractModel::BackboneProfiles;
    using BackboneProfile =  typename AbstractModel::BackboneProfile;

    using Histogram = typename AbstractModel::Histogram;

    using HeteroHistograms = typename AbstractModel::TransitionMatrices2D;

    static constexpr const char *LOADING = "loading";
    static constexpr const char *PREPROCESSING = "preprocessing";
    static constexpr const char *TRAINING = "training";
    static constexpr const char *CLASSIFICATION = "classification";

    using PriorityQueue = typename MatchSet<Score>::Queue<std::string_view>;
    using LeaderBoard = ClassificationCandidates<Score>;

public:
    Pipeline( ModelGenerator<States> modelTrainer,
              SimilarityFunctor<Histogram> similarity,
              Order order )
            : _modelGenerator( std::move( modelTrainer )),
              _similarity( std::move( similarity )),
              _order( order )
    {}

public:

    template<typename InputSequence>
    static std::vector<std::string>
    reducedAlphabetEntries( const std::vector<InputSequence> &entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( entries );
    }

    template<typename InputSequence>
    static std::map<std::string_view, std::vector<std::string >>
    reducedAlphabetEntries( const std::map<std::string_view, std::vector<InputSequence >> &entries )
    {
        std::map<std::string_view, std::vector<std::string >> newEntries;
        for (auto &[label, sequences] : entries)
            newEntries.emplace( label, reducedAlphabetEntries( sequences ));
        return newEntries;
    }

    template<typename TrainingDataType>
    std::vector<ScoredLabels>
    scoredPredictions( const std::vector<std::string> &queries,
                       const BackboneProfiles &backbones,
                       const BackboneProfiles &backgrounds,
                       const BackboneProfile &centralBackground,
                       const TrainingDataType &trainingClusters,
                       const ClassificationEnum classificationStrategy,
                       std::optional<std::reference_wrapper<const Selection>> selection = std::nullopt ) const
    {
        SVMConfiguration svmConfiguration;
        PCAConfiguration pcaConfiguration;
        LDAConfiguration ldaConfiguration;
        svmConfiguration.tuning = SVMConfiguration::defaultTuning();

        switch (classificationStrategy)
        {
            using MacroScoringEnum = typename MicroSimilarityBasedClassifier<States>::MacroScoringEnum;
            case ClassificationEnum::MacroSimilarityCombined:
            {
                auto model = MicroSimilarityBasedClassifier<States>(
                        MacroScoringEnum::Combine, _modelGenerator, _similarity );
                return model.scoredPredictions( reducedAlphabetEntries( queries ),
                                                backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::MacroSimilarityAccumulative :
            {
                auto model = MicroSimilarityBasedClassifier<States>(
                        MacroScoringEnum::Accumulative, _modelGenerator, _similarity );
                return model.scoredPredictions( reducedAlphabetEntries( queries ),
                                                backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::MacroSimilarityVoting :
            {
                auto model = MicroSimilarityBasedClassifier<States>(
                        MacroScoringEnum::Voting, _modelGenerator, _similarity );
                return model.scoredPredictions( reducedAlphabetEntries( queries ),
                                                backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::Propensity :
            {
                MCPropensityClassifier<States> classifier( _modelGenerator );
                return classifier.scoredPredictions( reducedAlphabetEntries( queries ),
                                                     backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::Segmentation :
            {
                MCSegmentationClassifier<States> classifier( backbones, backgrounds,
                                                             reducedAlphabetEntries( trainingClusters ),
                                                             _modelGenerator );
                return classifier.scoredPredictions( reducedAlphabetEntries( queries ));
            }
            case ClassificationEnum::SVM :
            {
                SVMCMMicroSimilarity<States> svm(
                        svmConfiguration, _modelGenerator, _similarity,
                        ldaConfiguration, pcaConfiguration );
                svm.fit( reducedAlphabetEntries( trainingClusters ),
                         backbones, backgrounds, centralBackground );
                return svm.scoredPredictions( reducedAlphabetEntries( queries ),
                                              backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::KNN :
            {
                KNNMCMicroSimilarity<States> knn(
                        7, _modelGenerator, _similarity, ldaConfiguration, pcaConfiguration );

                knn.fit( reducedAlphabetEntries( trainingClusters ),
                         backbones, backgrounds, centralBackground );
                return knn.scoredPredictions( reducedAlphabetEntries( queries ),
                                              backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::SVM_Stack :
            {
                SVMStackedMC<States> svm(
                        svmConfiguration, _modelGenerator, ldaConfiguration, pcaConfiguration  );

                svm.initWeakModels( _similarity );

                svm.fit( reducedAlphabetEntries( trainingClusters ),
                         backbones, backgrounds, centralBackground );
                return svm.scoredPredictions( reducedAlphabetEntries( queries ),
                                              backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::KNN_Stack :
            {
                KNNStackedMC<States> knn(
                        7, _modelGenerator, ldaConfiguration, pcaConfiguration );

                knn.initWeakModels( _similarity );

                knn.fit( reducedAlphabetEntries( trainingClusters ),
                         backbones, backgrounds, centralBackground );
                return knn.scoredPredictions( reducedAlphabetEntries( queries ),
                                              backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::KMERS :
            {
                MCKmersClassifier<States> classifier( _modelGenerator );
                return classifier.scoredPredictions( reducedAlphabetEntries( queries ),
                                                     backbones, backgrounds, centralBackground );
            }
            case ClassificationEnum::DiscretizedScales :
            {
                MCDiscretizedScalesClassifier<States> classifier( _modelGenerator, 15 );
                classifier.runTraining( trainingClusters );
                return classifier.scoredPredictions( queries );
            }
            default:
                throw std::runtime_error( "Undefined Strategy" );
        }
    }

    BackboneProfile
    balancedCentroid( const std::map<std::string_view, std::vector<std::string >> &trainingClusters ) const
    {
        auto centralBackground = _modelGenerator();
        for (auto &&[_, subset] : AbstractMC<States>::undersampleBalancing( trainingClusters ))
            centralBackground->addSequences( subset );
        centralBackground->normalize();
        return centralBackground;
    }

    void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries, const size_t k,
                                 const std::vector<std::string> &classificationStrategy )
    {
        std::set<std::string> classifiers;
        for (const auto &classifier : classificationStrategy)
            classifiers.insert( classifier );

        std::set<std::string> labels;
        for (const auto &entry : entries)
            labels.emplace( entry.label());
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
                                            return std::make_pair( p.first, std::string( p.second.sequence()));
                                        } );
                        return foldSequences;
                    } );
            return fSequences;
        };

        std::set<std::string_view> uniqueIds;
        for (auto &e : entries)
            uniqueIds.insert( e.memberId());

        fmt::print( "[All Sequences:{} (unique:{})]\n", entries.size(), uniqueIds.size());

        auto groupedEntries = LabeledEntry::groupEntriesByLabels( std::move( entries ));
        auto averageLength = LabeledEntry::groupAveragedValue<LabeledEntry>(
                groupedEntries,
                []( std::string_view, const auto &sequence ) -> double {
                    return sequence.length();
                } );

        auto varianceLength = LabeledEntry::groupAveragedValue<LabeledEntry>(
                groupedEntries,
                [&]( std::string_view label, const auto &sequence ) -> double {
                    double deviation = sequence.length() - averageLength.at( std::string( label ));
                    return deviation * deviation;
                } );

        auto labelsInfo = keys( groupedEntries );
        for (auto &l : labelsInfo)
            l = fmt::format( "{}({}|{})", l,
                             groupedEntries.at( l ).size(),
                             averageLength.at( l ));

        fmt::print( "[Clusters:{}][{}]\n",
                    groupedEntries.size(),
                    io::join( labelsInfo, "|" ));

        const Folds folds = kFoldStratifiedSplit( std::move( groupedEntries ), k );
        const FoldsSequences sFolds = extractSequences( folds );

        auto unzip = []( const std::vector<std::pair<std::string, LabeledEntry >> &items ) {
            std::vector<std::string_view> ids;
            std::vector<std::string> sequences;
            std::vector<std::string_view> ls;
            for (const auto &item : items)
            {
                ls.push_back( item.first );
                sequences.emplace_back( item.second.sequence());
                ids.push_back( item.second.memberId());
            }
            return std::make_tuple( ids, sequences, ls );
        };

        std::map<std::string, CrossValidationStatistics<std::string_view >> validation;
        EnsembleCrossValidation<std::string_view> ensembleValidation( folds );

        for (auto &classifier : classifiers)
            validation[classifier] = CrossValidationStatistics( k, viewLabels );

        for (size_t i = 0; i < k; ++i)
        {
            fmt::print( "Fold#{}:\n", i + 1 );
            const auto trainingClusters = joinFoldsExceptK( sFolds, i );
            const auto
            [ids, queries, qLabels] = unzip( folds.at( i ));
            fmt::print( "Training..\n" );
            const auto trainingData = AbstractModel::oversampleStateBalancing( trainingClusters );

            auto labelsAverageStates = LabeledEntry::groupAveragedValue<std::string_view>(
                    trainingData,
                    []( std::string_view, auto &&sequence ) -> double {
                        return sequence.length();
                    } );

            auto currentLabelsInfo = keys( trainingData, []( auto &&s ) -> std::string {
                return std::string( s );
            } );

            for (auto &l : currentLabelsInfo)
                l = fmt::format( "{}({}*{}={})", l,
                                 trainingData.at( l ).size(),
                                 labelsAverageStates.at( l ),
                                 labelsAverageStates.at( l ) * trainingData.at( l ).size());

            fmt::print( "[Clusters:{}][{}]\n",
                        trainingData.size(),
                        io::join( currentLabelsInfo, "|" ));

            auto reducedTrainingData = reducedAlphabetEntries( trainingData );
            BackboneProfiles backbones = AbstractModel::train( reducedTrainingData, _modelGenerator );
            BackboneProfiles backgrounds = AbstractModel::backgroundProfiles( reducedTrainingData, _modelGenerator );
            auto balancedBackgroundCentroid = balancedCentroid( reducedTrainingData );
            fmt::print( "[DONE] Training..\n" );

            fmt::print( "Classification..\n" );
            for (auto &classifier : classifiers)
            {
                auto classifierEnum = ClassifierEnum.at( classifier );
                auto predictions = scoredPredictions( queries, backbones, backgrounds,
                                                      balancedBackgroundCentroid,
                                                      trainingData, classifierEnum );

                assert( predictions.size() == qLabels.size() && qLabels.size() == queries.size() &&
                        queries.size() == ids.size());

                auto &cValidation = validation[classifier];
                for (size_t proteinIdx = 0; proteinIdx < queries.size(); ++proteinIdx)
                {
                    const auto &fold = folds.at( i );

                    const auto &id = fold.at( proteinIdx ).second.memberId();
                    const auto &label = fold.at( proteinIdx ).first;
                    const auto &prediction = predictions.at( proteinIdx );

                    cValidation.countInstance( i, prediction.top()->get().label(), label );
                    ensembleValidation.countInstance( i, classifier, id, prediction );
                }
            }
            fmt::print( "[DONE] Classification..\n" );
        }

        for (auto &[classifier, cvalidation] : validation)
        {
            fmt::print( "{{{}}} Cross-validation\n", classifier );
            cvalidation.printReport();
            cvalidation.printPerClassReport();
        }

//        for ( auto & [ensemble, cv, aucs] : ensembleValidation.majorityVotingOverallAccuracy())
//        {
//            fmt::print( "Majority MacroSimilarityVoting {{{}}} Cross-validation\n", io::join( ensemble, "," ));
//            cv.printReport();
//            for ( auto & [feature, auc] : aucs )
//                fmt::print( "AUC({}):{}\n", feature, auc.auc());
//        }
//
//        for ( auto & [ensemble, cv, aucs] : ensembleValidation.weightedVotingOverallAccuracy())
//        {
//            fmt::print( "Weighted MacroSimilarityVoting {{{}}} Cross-validation\n", io::join( ensemble, "," ));
//            cv.printReport();
//            for ( auto & [feature, auc] : aucs )
//                fmt::print( "AUC({}):{}\n", feature, auc.auc());
//        }
    }

private:
    const ModelGenerator<States> _modelGenerator;
    const SimilarityFunctor<Histogram> _similarity;
    const Order _order;
};

using PipelineVariant = MakeVariantType<Pipeline, SupportedAAGrouping>;

template<typename AAGrouping, typename Similarity>
PipelineVariant getConfiguredPipeline( MCModelsEnum model, Order mxOrder, Similarity similarity )
{
    static constexpr auto States = AAGrouping::StatesN;
    using MG = ModelGenerator<States>;
    using RMC = MC<States>;
    using ZMC = ZYMC<States>;
    using GMC = GappedMC<States>;
    using BMC = RegularizedBinaryMC<States, RMC>;
    using VMC = RegularizedVectorsMC<States, RMC>;

    switch (model)
    {
        case MCModelsEnum::RegularMC :
            return Pipeline<AAGrouping>(
                    MG::template create<RMC>( mxOrder ), similarity, mxOrder );
        case MCModelsEnum::ZhengYuanMC :
            return Pipeline<AAGrouping>(
                    MG::template create<ZMC>( mxOrder ), similarity, mxOrder );
        case MCModelsEnum::GappedMC :
            return Pipeline<AAGrouping>(
                    MG::template create<GMC>( mxOrder ), similarity, mxOrder );
        case MCModelsEnum::RegularizedBinaryMC :
            return Pipeline<AAGrouping>(
                    MG::template create<BMC>( mxOrder ), similarity, mxOrder );
        case MCModelsEnum::RegularizedVectorsMC :
            return Pipeline<AAGrouping>(
                    MG::template create<VMC>( mxOrder ), similarity, mxOrder );
        default:
            throw std::runtime_error( "Undefined Strategy" );
    }
};


template<typename AAGrouping>
PipelineVariant getConfiguredPipeline( CriteriaEnum criteria, MCModelsEnum model, Order mxOrder )
{
    static constexpr auto States = AAGrouping::StatesN;

    using AbstractModel = AbstractMC<States>;
    using Histogram = typename AbstractModel::Histogram;

    switch (criteria)
    {
        case CriteriaEnum::ChiSquared :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, ChiSquared::similarityFunctor<Histogram>());
        case CriteriaEnum::Cosine :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, Cosine::similarityFunctor<Histogram>());
        case CriteriaEnum::KullbackLeiblerDiv:
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, KullbackLeiblerDivergence::similarityFunctor<Histogram>());
        case CriteriaEnum::Gaussian :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, Gaussian::similarityFunctor<Histogram>());
        case CriteriaEnum::Intersection :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, Intersection::similarityFunctor<Histogram>());
        case CriteriaEnum::DensityPowerDivergence1 :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, DensityPowerDivergence1::similarityFunctor<Histogram>());
        case CriteriaEnum::DensityPowerDivergence2 :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, DensityPowerDivergence2::similarityFunctor<Histogram>());
        case CriteriaEnum::DensityPowerDivergence3:
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, DensityPowerDivergence3::similarityFunctor<Histogram>());
        case CriteriaEnum::ItakuraSaitu :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, ItakuraSaitu::similarityFunctor<Histogram>());
        case CriteriaEnum::Bhattacharyya :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, Bhattacharyya::similarityFunctor<Histogram>());
        case CriteriaEnum::Hellinger :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, Hellinger::similarityFunctor<Histogram>());
        case CriteriaEnum::MaxIntersection :
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, MaxIntersection::similarityFunctor<Histogram>());
        case CriteriaEnum::DWCosine:
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, DWCosine::similarityFunctor<Histogram>());
        case CriteriaEnum::Mahalanobis:
            return getConfiguredPipeline<AAGrouping>(
                    model, mxOrder, Mahalanobis::similarityFunctor<Histogram>());
        default:
            throw std::runtime_error( "Undefined Strategy" );
    }
};


PipelineVariant
getConfiguredPipeline( AminoAcidGroupingEnum grouping, CriteriaEnum criteria, MCModelsEnum model, Order mxOrder )
{
    switch (grouping)
    {
        case AminoAcidGroupingEnum::NoGrouping22:
            return getConfiguredPipeline<AAGrouping_NOGROUPING22>( criteria, model, mxOrder );
//            case AminoAcidGroupingEnum::DIAMOND11 :
//                return getConfiguredPipeline<AAGrouping_DIAMOND11>( criteria, model, mnOrder, mxOrder );
//            case AminoAcidGroupingEnum::OFER8 :
//                return getConfiguredPipeline<AAGrouping_OFER8>( criteria, model, mnOrder, mxOrder );
        case AminoAcidGroupingEnum::OFER15 :
            return getConfiguredPipeline<AAGrouping_OFER15>( criteria, model, mxOrder );
        default:
            throw std::runtime_error( "Undefined Grouping" );

    }
}

PipelineVariant getConfiguredPipeline( const std::string &groupingName,
                                       const std::string &criteria,
                                       const std::string &model, Order mxOrder )
{
    const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
    const CriteriaEnum criteriaLabel = CriteriaLabels.at( criteria );
    const MCModelsEnum modelLabel = MCModelLabels.at( model );

    return getConfiguredPipeline( groupingLabel, criteriaLabel, modelLabel, mxOrder );
}

}
#endif //MARKOVIAN_FEATURES_CONFIGUREDPIPELINE_HPP
