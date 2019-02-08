//
// Created by asem on 18/09/18.
//

#ifndef MARKOVIAN_FEATURES_ABSTRACTCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_ABSTRACTCLASSIFIER_HPP

#include "SimilarityMetrics.hpp"
#include "MCDefs.h"
#include "AbstractMC.hpp"

namespace MC {

enum class ClassificationEnum
{
    MacroSimilarityVoting,
    MacroSimilarityAccumulative,
    MacroSimilarityCombined,
    Propensity,
    Segmentation,
    SVM_MCParameters,
    KNN_MCParameters,
    SVM_MCSimilarity,
    KNN_MCSimilirity,
    KMERS,
    SVM_Stack,
    KNN_Stack,
    DiscretizedScales
};

static const std::map<std::string, ClassificationEnum> ClassifierEnum = {
        {"voting",       ClassificationEnum::MacroSimilarityVoting},
        {"acc",          ClassificationEnum::MacroSimilarityAccumulative},
        {"propensity",   ClassificationEnum::Propensity},
        {"segmentation", ClassificationEnum::Segmentation},
        {"svm_mcp",      ClassificationEnum::SVM_MCParameters},
        {"knn_mcp",      ClassificationEnum::KNN_MCParameters},
        {"svm_mcs",      ClassificationEnum::SVM_MCSimilarity},
        {"knn_mcs",      ClassificationEnum::KNN_MCSimilirity},
        {"svm_stack",    ClassificationEnum::SVM_Stack},
        {"knn_stack",    ClassificationEnum::KNN_Stack},
        {"kmers",        ClassificationEnum::KMERS},
        {"discretized",  ClassificationEnum::DiscretizedScales},
};

static const std::map<ClassificationEnum, std::string_view> ClassifierLabel = []() {
    std::map<ClassificationEnum, std::string_view> m;
    for (auto &[label, enumm] : ClassifierEnum)
        m.emplace( enumm, label );
    return m;
}();

template<size_t States>
class AbstractMCClassifier
{
public:
    using BackboneProfile = typename AbstractMC<States>::BackboneProfile;
    using BackboneProfiles = typename AbstractMC<States>::BackboneProfiles;

    virtual ~AbstractMCClassifier() = default;

    explicit AbstractMCClassifier( ModelGenerator <States> generator )
            : _generator( generator )
    {
    }

    template<typename TrainingDataType>
    void trainMC( const TrainingDataType &trainingData )
    {
        for (auto &&[label, classSequences]: trainingData)
        {
            auto backboneIt = _backbones.emplace( label, _generator()).first;
            auto &&backbone = backboneIt->second;
            backbone.addSequences( classSequences );

            auto backgroundIt = _backgrounds.emplace( label, _generator()).first;
            auto &&background = backgroundIt->second;
            for (auto &&[backgroundLabel, backgroundSequences] : trainingData)
            {
                if ( backgroundLabel != label )
                {
                    background.addSequences( backgroundSequences );
                }
            }
            backbone.normalize();
            background.normalize();
            _centralBackground = _generator();
            for (auto &&[_, subset] : AbstractMC<States>::undersampleBalancing( trainingData ))
                _centralBackground->addSequences( subset );
            _centralBackground->normalize();
        }
    }

public:
    template<typename SequencesType>
    std::vector<std::string_view> predict( const SequencesType &sequences ) const
    {
        return predict( sequences, _backbones, _backgrounds, _centralBackground );
    }

    template<typename SequencesType>
    std::vector<std::string_view> predict(
            const SequencesType &sequences,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const std::optional<BackboneProfile> &centralBackground = std::nullopt
    ) const
    {
        assert( _validTraining( backboneProfiles, backgroundProfiles, centralBackground ));
        std::vector<std::string_view> labels;
        for (auto &&seq : sequences)
            labels.emplace_back( _bestPrediction( seq, backboneProfiles, backgroundProfiles, centralBackground ));

        return labels;
    }

    template<typename InputSequence>
    std::vector<ScoredLabels> scoredPredictions(
            const std::vector<InputSequence> &sequences
    ) const
    {
        return scoredPredictions( sequences, _backbones, _backgrounds, _centralBackground );
    }

    template<typename InputSequence>
    std::vector<ScoredLabels> scoredPredictions(
            const std::vector<InputSequence> &sequences,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const
    {
        assert( _validTraining( backboneProfiles, backgroundProfiles, centralBackground ));
        std::vector<ScoredLabels> scoredLabels;
        for (auto &&seq : sequences)
            scoredLabels.emplace_back(
                    scoredPredictions( seq, backboneProfiles, backgroundProfiles, centralBackground ));

        return scoredLabels;
    }

    ScoredLabels scoredPredictions(
            std::string_view sequence
    ) const
    {
        assert( _validTraining( _backbones, _backgrounds, _centralBackground ));
        return _predict( sequence, _backbones, _backgrounds, _centralBackground );
    }

    ScoredLabels scoredPredictions(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const
    {
        assert( _validTraining( backboneProfiles, backgroundProfiles, centralBackground ));
        return _predict( sequence, backboneProfiles, backgroundProfiles, centralBackground );
    }

protected:
    static bool _validTraining(
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    )
    {
        return !backgroundProfiles.empty() &&
               backboneProfiles.size() == backgroundProfiles.size();
    };

    bool _validTraining() const
    {
        return _validTraining( _backbones, _backgrounds, _centralBackground );
    }

    std::string_view _bestPrediction(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const
    {
        auto predictions = _predict( sequence, backgroundProfiles, backgroundProfiles, centralBackground );
        if ( auto top = predictions.top(); top )
        {
            return top->get().label();
        } else return unclassified;
    }

    virtual ScoredLabels _predict(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const = 0;

protected:
    const ModelGenerator <States> _generator;
    BackboneProfiles _backbones;
    BackboneProfiles _backgrounds;
    BackboneProfile _centralBackground;
};

template<size_t States>
class MCPropensityClassifier : public AbstractMCClassifier<States>
{
    using MCModel = AbstractMC<States>;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using ScoringFunction = std::function<double( std::string_view )>;
    using BackboneProfile = typename MCModel::BackboneProfile;

public:
    virtual ~MCPropensityClassifier() = default;

    MCPropensityClassifier( ModelGenerator <States> generator )
            : AbstractMCClassifier<States>( generator )
    {}

protected:
    ScoredLabels _predict(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &
    ) const override
    {
        std::map<std::string_view, double> propensitites;

        for (auto&[label, backbone] :backboneProfiles)
        {
            auto &bg = backgroundProfiles.at( label );
            double logOdd = backbone->propensity( sequence ) - bg->propensity( sequence );
            propensitites[label] = logOdd;
        }

        propensitites = minmaxNormalize( std::move( propensitites ));

        ScoredLabels matchSet( backboneProfiles.size());
        for (auto &[label, relativeAffinity] : propensitites)
            matchSet.emplace( label, relativeAffinity );

        return matchSet;
    }
};

template<size_t States>
class MicroSimilarityBasedClassifier : public AbstractMCClassifier<States>
{
public:
    enum class MacroScoringEnum
    {
        Accumulative,
        Voting,
        Combine
    };

protected:
    using Model = AbstractMC<States>;
    using Histogram = typename Model::Histogram;
    using TransitionMatrices2D = typename Model::TransitionMatrices2D;
    using BackboneProfile = typename Model::BackboneProfile;
    using BackboneProfiles = typename Model::BackboneProfiles;
    using Similarity = SimilarityFunctor<Histogram>;
    using MicroMeasurements = std::map<std::string_view, std::unordered_map<Order, std::unordered_map<HistogramID, double >>>;
    using AlternativeMeasurements = std::unordered_map<Order, std::unordered_map<HistogramID, double >>;
public:
    explicit MicroSimilarityBasedClassifier(
            const MacroScoringEnum macroScoring,
            const ModelGenerator <States> &generator,
            const SimilarityFunctor<Histogram> similarityFunctor
    )
            : AbstractMCClassifier<States>( generator ), _similarityFunctor( similarityFunctor ),
              _macroScoring( macroScoring )
    {}

    virtual ~MicroSimilarityBasedClassifier() = default;

protected:
    template<typename MicroMeasurementsType, typename AlternativeMeasurementsType>
    static std::map<std::string_view, double>
    macroScoresFromMicroMeasurement(
            MicroMeasurementsType &&microMeasurements,
            AlternativeMeasurementsType &&alternativeMeasurements,
            const SimilarityFunctor<Histogram> &similarityFunctor
    )
    {
        std::map<std::string_view, double> macro;
        for (auto &&[label, measurements] : microMeasurements)
        {
            double &labelMacro = macro[label];
            for (auto &&[order, isoMeasurements] : measurements)
            {
                auto &isoAlternativeMeasurements = alternativeMeasurements.at( order );
                labelMacro = std::accumulate(
                        isoMeasurements.cbegin(), isoMeasurements.cend(), labelMacro,
                        [&](
                                double acc,
                                std::pair<HistogramID, double> &&value
                        ) {
                            if ( double measurement = value.second; !std::isnan( measurement ))
                            {
                                return acc + measurement;
                            } else
                            {
                                HistogramID id = value.first;
                                return acc + isoAlternativeMeasurements.at( id );
                            }
                        } );
            }
        }
        if ( similarityFunctor.cost )
        {
            for (auto &[label, cost] : macro)
                cost *= -1;
        }
        return macro;
    }

    template<typename MicroMeasurementsType>
    static std::map<std::string_view, double>
    votingFromMicroMeasurements(
            MicroMeasurementsType &&microMeasurements,
            std::function<bool(
                    double,
                    double
            )> closerThan
    )
    {
        std::map<std::string_view, double> votes;
        std::unordered_map<Order, std::unordered_map<HistogramID, std::pair<std::string_view, double >>> closest;
        for (auto &&[label, measurements] : microMeasurements)
        {
            for (auto &&[order, isoMeasurements] : measurements)
            {
                for (auto &&[id, measurement] : isoMeasurements)
                {
                    auto current = std::make_pair( label, measurement );
                    auto[nearestIt, insertionResult] = closest[order].try_emplace( id, current );
                    if ( !insertionResult )
                    {
                        auto &&currentNearest = nearestIt->second;
                        if ( closerThan( measurement, currentNearest.second ))
                        {
                            currentNearest.swap( current );
                        }
                    }
                }
            }
        }

        for (auto &&[_, nearestHistograms] : closest)
            for (auto &&[_, nearestHistogram] : nearestHistograms)
                ++votes[nearestHistogram.first];
        return votes;
    }

    ScoredLabels _predict(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const override
    {
        MicroMeasurements measurements;
        AlternativeMeasurements alternatives;

        auto closerThan = _similarityFunctor.closerThan;
        auto bestInfinity = _similarityFunctor.best;
        auto model = this->_generator();
        model->train( sequence );

        auto histograms = std::move( model->stealCentroids());
        for (const auto &[label, backbone] : backboneProfiles)
        {
            auto &_measurements = measurements[label];
            histograms.forEach(
                    [&](
                            Order order,
                            HistogramID id,
                            const Histogram &histogram
                    ) {
                        double &measurement = _measurements[order][id];
                        double &furthest = alternatives[order].try_emplace( id, bestInfinity ).first->second;

                        auto backboneHistogram = backbone->centroid( order, id );
                        if ( backboneHistogram )
                        {
                            auto standardDeviation = backbone->standardDeviation( order, id );
                            measurement = _similarityFunctor( backboneHistogram->get(), histogram,
                                                              standardDeviation->get());

                        } else if ( centralBackground )
                        {
                            if ( auto center = centralBackground->centroid( order, id ); center.has_value())
                            {
                                auto standardDeviation = centralBackground->standardDeviation( order, id );
                                measurement = _similarityFunctor( center->get(), histogram,
                                                                  standardDeviation->get());
                            }
                        } else
                        {
                            measurement = nan;
                        }
                        if ( !std::isnan( measurement ))
                        {
                            if ( closerThan( furthest, measurement ))
                                furthest = measurement;
                        }
                    } );
        }

        ScoredLabels matchSet( backboneProfiles.size());

        switch (_macroScoring)
        {
            case MacroScoringEnum::Accumulative :
            {
                auto macroScores = minmaxNormalize(
                        macroScoresFromMicroMeasurement( measurements, alternatives, _similarityFunctor ));
                for (auto &[label, _] : backboneProfiles)
                    matchSet.emplace( label, macroScores[label] );
            }
                break;
            case MacroScoringEnum::Voting:
            {
                auto voteScores = minmaxNormalize( votingFromMicroMeasurements( measurements, closerThan ));
                for (auto &[label, _] : backboneProfiles)
                    matchSet.emplace( label, voteScores[label] );
            }
                break;
            case MacroScoringEnum::Combine :
            {
                auto macroScores = minmaxNormalize(
                        macroScoresFromMicroMeasurement( measurements, alternatives, _similarityFunctor ));
                auto voteScores = minmaxNormalize( votingFromMicroMeasurements( measurements, closerThan ));
                for (auto &[label, _] : backboneProfiles)
                    matchSet.emplace( label, (macroScores[label] + voteScores[label]) / 2 );
            }
                break;
            default:
            {
                throw std::runtime_error( "Unhandled macro scoring method." );
            }
        }

        return matchSet;
    }

private:
    const MacroScoringEnum _macroScoring;
    const SimilarityFunctor<Histogram> _similarityFunctor;
};

}
#endif //MARKOVIAN_FEATURES_ABSTRACTCLASSIFIER_HPP
