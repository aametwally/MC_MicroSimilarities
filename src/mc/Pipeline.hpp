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

#include "MicroSimilarityVotingClassifier.hpp"
#include "MacroSimilarityClassifier.hpp"
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
        Accumulative,
        Propensity,
        SVM,
        SVM_Propensity,
        KNN_Propensity,
        KMERS
    };

    static const std::map<std::string, ClassificationMethod> ClassificationMethodLabel = {
            {"voting",         ClassificationMethod::Voting},
            {"acc",            ClassificationMethod::Accumulative},
            {"propensity",     ClassificationMethod::Propensity},
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

        std::vector<std::pair<std::string_view, std::string_view >>
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
            std::vector<std::pair<std::string_view, std::string_view >> classifications;
            for (auto i = 0; i < queries.size(); ++i)
                classifications.emplace_back( trueLabels.at( i ), results.at( i ));

            return classifications;
        }


        std::vector<std::string_view>
        predict_KMERS(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            std::vector<std::string_view> results;
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
                if ( auto top = vPQ.top(); top )
                    results.emplace_back( top->get().getLabel());
                else results.emplace_back();
            }
            return results;
        }

        std::vector<std::string_view>
        predict_ACCUMULATIVE(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );
            auto model = MacroSimilarityClassifier<Grouping>( targets, background, selection,
                                                              _modelTrainer, _similarity );
            return model.predict( queries );
        }

        std::vector<std::string_view>
        predict_VOTING(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );
            auto model = MicroSimilarityVotingClassifier<Grouping>( targets, background, selection,
                                                                    _modelTrainer, _similarity );
            return model.predict( queries );
        }

        std::vector<std::string_view>
        predict_PROPENSITY(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );
            std::vector<std::string_view> predictions;
            for (auto &query : queries)
            {
                PriorityQueue matchSet( targets.size());
                for (auto&[label, backbone] :targets)
                {
                    auto &bg = background.at( label );
                    double logOdd = backbone->propensity( query ) - bg->propensity( query );
                    matchSet.emplace( label, logOdd );
                }
                if ( auto top = matchSet.top(); top )
                    predictions.emplace_back( top->get().getLabel());
                else predictions.emplace_back( unclassified );
            }
            return predictions;
        }

        std::vector<std::string_view>
        predict_SVM(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            SVMMCParameters<Grouping> svm( _modelTrainer );
            svm.fit( targets, background, trainingClusters );
            return svm.predict( queries );
        }

        std::vector<std::string_view>
        predict_SVM_Propensity(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            SVMConfusionMC<Grouping> svm( _modelTrainer );
            svm.fit( targets, background, trainingClusters );
            return svm.predict( queries );
        }

        std::vector<std::string_view>
        predict_KNN_Propensity(
                const std::vector<std::string> &queries,
                const BackboneProfiles &targets,
                const std::map<std::string, std::vector<std::string >> &trainingClusters,
                const Selection &selection ) const
        {
            BackboneProfiles background = backgroundProfiles( trainingClusters, selection );

            KNNConfusionMC<Grouping> knn( 3 );
            knn.fit( targets, background, trainingClusters );
            return knn.predict( queries );
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

        std::vector<std::string_view>
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
                case ClassificationMethod::Voting :
                    return predict_VOTING( queries, targets, trainingClusters, selection );
                case ClassificationMethod::Propensity :
                    return predict_PROPENSITY( queries, targets, trainingClusters, selection );
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

        void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries, const size_t k,
                                     const std::vector<std::string> &classificationStrategy )
        {

            std::set<std::string> classifiers;
            for (const auto &classifier : classificationStrategy)
                classifiers.insert( classifier );

            std::set<std::string> labels;
            std::vector<std::string> members;
            for (const auto &entry : entries)
            {
                labels.insert( entry.getLabel());
                members.push_back( entry.getMemberId());
            }


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




            using MemberAssignments = std::map<std::string_view,
                    std::map<std::string_view, std::vector<std::string_view >>>;

            MemberAssignments assignments;
            std::map<std::string, CrossValidationStatistics<std::string_view >> validation;

            for (auto &classifier : classifiers)
                validation[classifier] = CrossValidationStatistics( k, viewLabels );


            for (auto i = 0; i < k; ++i)
            {
                auto trainingClusters = joinFoldsExceptK( sFolds, i );
                const auto[ids, test, tLabels] = unzip( folds.at( i ));
                const auto[selection, filteredProfiles] = featureSelection( trainingClusters );

                for (auto &classifier : classifiers)
                {
                    auto classifierEnum = ClassificationMethodLabel.at( classifier );
                    auto predictions = classify_VALIDATION( test, tLabels, filteredProfiles,
                                                            trainingClusters, selection,
                                                            classifierEnum );

                    assert( predictions.size() == tLabels.size() && tLabels.size() == test.size());

                    auto &cValidation = validation[classifier];
                    for (auto[trueClass, prediction] : predictions)
                        cValidation.countInstance( i, prediction, trueClass );
                }
            }

            for (auto &classifier : classifiers)
            {
                validation[classifier].printReport( classifier );
                fmt::print( "\n" );
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
