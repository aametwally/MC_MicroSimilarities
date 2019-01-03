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
#include "ZYMC.hpp"
#include "GappedMC.hpp"
#include "MCFeatures.hpp"

#include "MCPropensityClassifier.hpp"
#include "MCDiscriminativeClassifier.hpp"
#include "MCSegmentationClassifier.hpp"
#include "MCKmersClassifier.hpp"
#include "MicroSimilarityVotingClassifier.hpp"
#include "MacroSimilarityClassifier.hpp"
#include "SVMMCParameters.hpp"
#include "KNNMCParameters.hpp"
#include "SVMConfusionMC.hpp"
#include "KNNConfusionMC.hpp"
#include "MCDiscretizedScalesClassifier.hpp"

#include "AAIndexClustering.hpp"
#include "SimilarityMetrics.hpp"
#include "MCSegmentationClassifier.hpp"


namespace MC
{

enum class MCModelsEnum
{
    RegularMC ,
    ZhengYuanMC ,
    GappedMC
};

const std::map<std::string , MCModelsEnum> MCModelLabels{
        {"rmc" , MCModelsEnum::RegularMC} ,
        {"zymc" , MCModelsEnum::ZhengYuanMC} ,
        {"gmc" , MCModelsEnum::GappedMC}
};

template < typename Grouping >
class Pipeline
{

private:
    static constexpr auto States = Grouping::StatesN;
    using MCF = MCFeatures<States>;

    using AbstractModel = AbstractMC<States>;

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
    Pipeline( ModelGenerator<States> modelTrainer , Similarity similarity )
            : _modelTrainer( modelTrainer ) ,
              _similarity( similarity )
    {

    }

public:

    template < typename Entries >
    static std::vector<LabeledEntry>
    reducedAlphabetEntries( Entries &&entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( std::forward<Entries>( entries ));
    }

    std::vector<ScoredLabels>
    scoredPredictions( const std::vector<std::string> &queries ,
                       const BackboneProfiles &backbones ,
                       const BackboneProfiles &background ,
                       const std::map<std::string_view , std::vector<std::string >> &trainingClusters ,
                       const ClassificationEnum classificationStrategy ,
                       std::optional<std::reference_wrapper<const Selection>> selection = std::nullopt ) const
    {
        switch ( classificationStrategy )
        {
        case ClassificationEnum::Accumulative :
        {
            auto model = MacroSimilarityClassifier<States>(
                    backbones , background , _modelTrainer , _similarity , selection );
            return model.scoredPredictions( queries );
        }
        case ClassificationEnum::Voting :
        {
            auto model = MicroSimilarityVotingClassifier<States>(
                    backbones , background , _modelTrainer , _similarity , selection );
            return model.scoredPredictions( queries );
        }
        case ClassificationEnum::Propensity :
        {
            MCPropensityClassifier<States> classifier( backbones , background );
            return classifier.scoredPredictions( queries );
        }
        case ClassificationEnum::DiscriminativePropensity :
        {
            MCDiscriminativeClassifier<States> classifier( backbones , background );
            return classifier.scoredPredictions( queries );
        }
        case ClassificationEnum::Segmentation :
        {
            MCSegmentationClassifier<States> classifier( backbones , background , trainingClusters ,
                                                         _modelTrainer );
            return classifier.scoredPredictions( queries );
        }
        case ClassificationEnum::SVM :
        {
            SVMMCParameters<States> svm( _modelTrainer , _similarity );
            svm.fit( backbones , background , trainingClusters );
            return svm.scoredPredictions( queries );
        }
        case ClassificationEnum::KNN :
        {
            KNNMCParameters<States> knn( 7 , _modelTrainer , _similarity );
            knn.fit( backbones , background , trainingClusters );
            return knn.scoredPredictions( queries );

        }
        case ClassificationEnum::SVM_Stack :
        {
            SVMConfusionMC<States> svm( _modelTrainer );
            svm.fit( backbones , background , trainingClusters , _modelTrainer , _similarity , selection );
            return svm.scoredPredictions( queries );
        }
        case ClassificationEnum::KNN_Stack :
        {
            KNNConfusionMC<States> knn( 7 );
            knn.fit( backbones , background , trainingClusters , _modelTrainer , _similarity , selection );
            return knn.scoredPredictions( queries );
        }
        case ClassificationEnum::KMERS :
        {
            MCKmersClassifier<States> classifier( backbones , background );
            return classifier.scoredPredictions( queries );
        }
        case ClassificationEnum::DiscretizedScales :
        {
            MCDiscretizedScalesClassifier<5> classifier( trainingClusters , 5 );
            classifier.runTraining();
            return classifier.scoredPredictions( queries );
        }
        default:throw std::runtime_error( "Undefined Strategy" );
        }
    }

    void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries , const size_t k ,
                                 const std::vector<std::string> &classificationStrategy )
    {
        std::set<std::string> classifiers;
        for ( const auto &classifier : classificationStrategy )
            classifiers.insert( classifier );

        std::set<std::string> labels;
        for ( const auto &entry : entries )
            labels.insert( entry.getLabel());
        auto viewLabels = std::set<std::string_view>( labels.cbegin() , labels.cend());

        using Fold = std::vector<std::pair<std::string , LabeledEntry >>;
        using FoldSequences = std::vector<std::pair<std::string , std::string >>;
        using Folds = std::vector<Fold>;
        using FoldsSequences = std::vector<FoldSequences>;

        auto extractSequences = []( const Folds &folds )
        {
            FoldsSequences fSequences;
            std::transform( folds.cbegin() , folds.cend() ,
                            std::back_inserter( fSequences ) , []( const Fold &f )
                            {
                                FoldSequences foldSequences;
                                std::transform( f.cbegin() , f.cend() , std::back_inserter( foldSequences ) ,
                                                []( const auto &p )
                                                {
                                                    return std::make_pair( p.first , p.second.getSequence());
                                                } );
                                return foldSequences;
                            } );
            return fSequences;
        };

        std::set<std::string_view> uniqueIds;
        for ( auto &e : entries )
            uniqueIds.insert( e.getMemberId());

        fmt::print( "[All Sequences:{} (unique:{})]\n" , entries.size() , uniqueIds.size());


        auto groupedEntries = LabeledEntry::groupEntriesByLabels( reducedAlphabetEntries( std::move( entries )));


        auto labelsInfo = keys( groupedEntries );
        for ( auto &l : labelsInfo ) l = fmt::format( "{}({})" , l , groupedEntries.at( l ).size());
        fmt::print( "[Clusters:{}][{}]\n" ,
                    groupedEntries.size() ,
                    io::join( labelsInfo , "|" ));

        const Folds folds = kFoldStratifiedSplit( std::move( groupedEntries ) , k );
        const FoldsSequences sFolds = extractSequences( folds );

        auto unzip = []( const std::vector<std::pair<std::string , LabeledEntry >> &items )
        {
            std::vector<std::string_view> ids;
            std::vector<std::string> sequences;
            std::vector<std::string_view> ls;
            for ( const auto &item : items )
            {
                ls.push_back( item.first );
                sequences.push_back( item.second.getSequence());
                ids.push_back( item.second.getMemberId());
            }
            return std::make_tuple( ids , sequences , ls );
        };

        std::map<std::string , CrossValidationStatistics<std::string_view >> validation;
        EnsembleCrossValidation<std::string_view> ensembleValidation( folds );

        for ( auto &classifier : classifiers )
            validation[classifier] = CrossValidationStatistics( k , viewLabels );


        for ( size_t i = 0; i < k; ++i )
        {
            fmt::print( "Fold#{}:\n" , i + 1 );
            auto trainingClusters = joinFoldsExceptK( sFolds , i );
            const auto
            [ids , queries , qLabels] = unzip( folds.at( i ));
            fmt::print( "Training..\n" );
            BackboneProfiles backbones = AbstractModel::train( trainingClusters , _modelTrainer );
            BackboneProfiles backgrounds = AbstractModel::backgroundProfiles( trainingClusters , _modelTrainer );
            fmt::print( "[DONE] Training..\n" );

            fmt::print( "Classification..\n" );
            for ( auto &classifier : classifiers )
            {
                auto classifierEnum = ClassifierEnum.at( classifier );
                auto predictions = scoredPredictions( queries , backbones , backgrounds ,
                                                      trainingClusters , classifierEnum );

                assert( predictions.size() == qLabels.size() && qLabels.size() == queries.size() &&
                        queries.size() == ids.size());

                auto &cValidation = validation[classifier];
                for ( size_t proteinIdx = 0; proteinIdx < queries.size(); ++proteinIdx )
                {
                    const auto &fold = folds.at( i );

                    const auto &id = fold.at( proteinIdx ).second.getMemberId();
                    const auto &label = fold.at( proteinIdx ).first;
                    const auto &prediction = predictions.at( proteinIdx );

                    cValidation.countInstance( i , prediction.top()->get().getLabel() , label );
                    ensembleValidation.countInstance( i , classifier , id , prediction );
                }
            }
            fmt::print( "[DONE] Classification..\n" );

        }

        for ( auto &[classifier , cvalidation] : validation )
        {
            fmt::print( "{{{}}} Cross-validation\n" , classifier );
            cvalidation.printReport();
            cvalidation.printPerClassReport();
        }

//        for ( auto & [ensemble, cv, aucs] : ensembleValidation.majorityVotingOverallAccuracy())
//        {
//            fmt::print( "Majority Voting {{{}}} Cross-validation\n", io::join( ensemble, "," ));
//            cv.printReport();
//            for ( auto & [feature, auc] : aucs )
//                fmt::print( "AUC({}):{}\n", feature, auc.auc());
//        }
//
//        for ( auto & [ensemble, cv, aucs] : ensembleValidation.weightedVotingOverallAccuracy())
//        {
//            fmt::print( "Weighted Voting {{{}}} Cross-validation\n", io::join( ensemble, "," ));
//            cv.printReport();
//            for ( auto & [feature, auc] : aucs )
//                fmt::print( "AUC({}):{}\n", feature, auc.auc());
//        }
    }

private:
    const ModelGenerator<States> _modelTrainer;
    const Similarity _similarity;
};

using PipelineVariant = MakeVariantType<Pipeline , SupportedAAGrouping>;

template < typename AAGrouping , typename Similarity >
PipelineVariant getConfiguredPipeline( MCModelsEnum model , Order mxOrder , Similarity similarity )
{
    static constexpr auto States = AAGrouping::StatesN;
    using MG = ModelGenerator<States>;
    using RMC = MC<States>;
    using ZMC = ZYMC<States>;
    using GMC = GappedMC<States>;

    switch ( model )
    {
    case MCModelsEnum::RegularMC :return Pipeline<AAGrouping>( MG::template create<RMC>( mxOrder ) , similarity );
    case MCModelsEnum::ZhengYuanMC :return Pipeline<AAGrouping>( MG::template create<ZMC>( mxOrder ) , similarity );
    case MCModelsEnum::GappedMC :return Pipeline<AAGrouping>( MG::template create<GMC>( mxOrder ) , similarity );
    default:throw std::runtime_error( "Undefined Strategy" );
    }
};


template < typename AAGrouping >
PipelineVariant getConfiguredPipeline( CriteriaEnum criteria , MCModelsEnum model , Order mxOrder )
{
    static constexpr auto States = AAGrouping::StatesN;

    using AbstractModel = AbstractMC<States>;
    using Histogram = typename AbstractModel::Histogram;

    switch ( criteria )
    {
    case CriteriaEnum::ChiSquared :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , ChiSquared::function<Histogram> );
    case CriteriaEnum::Cosine :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , Cosine::function<Histogram> );
    case CriteriaEnum::KullbackLeiblerDiv:
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , KullbackLeiblerDivergence::function<Histogram> );
    case CriteriaEnum::Gaussian :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , Gaussian::function<Histogram> );
    case CriteriaEnum::Intersection :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , Intersection::function<Histogram> );
    case CriteriaEnum::DensityPowerDivergence1 :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , DensityPowerDivergence1::function<Histogram> );
    case CriteriaEnum::DensityPowerDivergence2 :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , DensityPowerDivergence2::function<Histogram> );
    case CriteriaEnum::DensityPowerDivergence3:
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , DensityPowerDivergence3::function<Histogram> );
    case CriteriaEnum::ItakuraSaitu :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , ItakuraSaitu::function<Histogram> );
    case CriteriaEnum::Bhattacharyya :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , Bhattacharyya::function<Histogram> );
    case CriteriaEnum::Hellinger :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , Hellinger::function<Histogram> );
    case CriteriaEnum::MaxIntersection :
        return getConfiguredPipeline<AAGrouping>(
                model , mxOrder , MaxIntersection::function<Histogram> );
    default:throw std::runtime_error( "Undefined Strategy" );
    }
};


PipelineVariant
getConfiguredPipeline( AminoAcidGroupingEnum grouping , CriteriaEnum criteria , MCModelsEnum model , Order mxOrder )
{
    switch ( grouping )
    {
    case AminoAcidGroupingEnum::NoGrouping22:
        return getConfiguredPipeline<AAGrouping_NOGROUPING22>( criteria , model , mxOrder );
//            case AminoAcidGroupingEnum::DIAMOND11 :
//                return getConfiguredPipeline<AAGrouping_DIAMOND11>( criteria, model, mnOrder, mxOrder );
//            case AminoAcidGroupingEnum::OFER8 :
//                return getConfiguredPipeline<AAGrouping_OFER8>( criteria, model, mnOrder, mxOrder );
    case AminoAcidGroupingEnum::OFER15 :return getConfiguredPipeline<AAGrouping_OFER15>( criteria , model , mxOrder );
    default:throw std::runtime_error( "Undefined Grouping" );

    }
}

PipelineVariant getConfiguredPipeline( const std::string &groupingName ,
                                       const std::string &criteria ,
                                       const std::string &model , Order mxOrder )
{
    const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
    const CriteriaEnum criteriaLabel = CriteriaLabels.at( criteria );
    const MCModelsEnum modelLabel = MCModelLabels.at( model );

    return getConfiguredPipeline( groupingLabel , criteriaLabel , modelLabel , mxOrder );
}

}
#endif //MARKOVIAN_FEATURES_CONFIGUREDPIPELINE_HPP
