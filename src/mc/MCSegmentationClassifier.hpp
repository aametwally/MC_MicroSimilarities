//
// Created by asem on 11/11/18.
//

#ifndef MARKOVIAN_FEATURES_MCSEGMENTATIONCLASSIFIER_H
#define MARKOVIAN_FEATURES_MCSEGMENTATIONCLASSIFIER_H

#include "AbstractClassifier.hpp"
#include "AbstractMC.hpp"
#include "SequenceAnnotator.hpp"

namespace MC {
    template<typename Grouping>
    class MCSegmentationClassifier : public AbstractClassifier
    {
        static constexpr size_t MAX_SEGMENTS = 10;
        using MCModel = AbstractMC<Grouping>;

        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using BackboneProfile = typename MCModel::BackboneProfile;

        using ScoreFunction = SequenceAnnotator::ScoreFunction;
    public:
        explicit MCSegmentationClassifier( const BackboneProfiles &backbones,
                                           const BackboneProfiles &backgrounds,
                                           const std::map<std::string_view, std::vector<std::string>> &trainingSequences,
                                           const ModelGenerator <Grouping> &modelTrainer )
                : _backbones( backbones ),
                  _background( backgrounds ),
                  _scoringFunctions( _extractScoringFunctions( backbones , backgrounds )),
                  _modelTrainer( modelTrainer )
        {
            _segmentationLearners =
                    _learnBySegmentation( backbones, backgrounds , _scoringFunctions ,
                            trainingSequences, modelTrainer );
        }

        virtual ~MCSegmentationClassifier() = default;

    protected:
        bool _validTraining() const override
        {
            return _backbones.size() == _background.size();
        }

        static std::vector<ScoreFunction>
        _extractScoringFunctions( const BackboneProfiles &profiles , const BackboneProfiles &backgrounds )
        {
            std::vector<ScoreFunction> scoringFunctions;
            for (auto &[l, profile] : profiles)
            {
                auto &background = backgrounds.at( l );
                scoringFunctions.emplace_back( [&]( std::string_view query ) -> double {
                    assert( !query.empty());
                    char state = query.back();
                    query.remove_suffix( 1 );
                    return profile->transitionalPropensity( query, state ) -
                           background->transitionalPropensity( query, state );
                } );
            }
            return scoringFunctions;
        }

        static std::vector<std::vector<double >>
        _extractBinaryScores( std::string_view query,
                              const BackboneProfile &profile,
                              const BackboneProfile &background )
        {
            std::vector<std::vector<double >> scoresForward;

            scoresForward.emplace_back( profile->compensatedPropensityVector( query ));
            scoresForward.emplace_back( background->compensatedPropensityVector( query ));

            return scoresForward;
        }

        static BackboneProfiles
        _learnBySegmentation( const BackboneProfiles &backbones,
                              const BackboneProfiles &backgrounds,
                              const std::vector< ScoreFunction > &scoringFunctions,
                              const std::map<std::string_view, std::vector<std::string>> &trainingSequences,
                              const ModelGenerator <Grouping> &modelTrainer )
        {
            std::vector<std::string_view> labels;
            std::transform( backbones.cbegin(), backbones.cend(), std::back_inserter( labels ),
                            []( const auto &p ) { return p.first; } );

            std::map<std::string_view, std::vector<std::string_view >> trainingSegments;
            for (auto &[l, sequences] : trainingSequences)
            {
                auto &segments = trainingSegments[l];
                for (auto &sequence : sequences)
                {
                    SequenceAnnotator annotator( sequence, scoringFunctions );
                    std::vector<SequenceAnnotation> annotations = annotator.annotate( MAX_SEGMENTS );

//                    for( auto &annotation : annotations )
//                    {
//                        for (auto &segment : annotation.getSegments())
//                        {
//                            if ( auto segmentLabel = labels.at( segment.getLabel()); segmentLabel == l )
//                                segments.push_back( segment.getSubsequence());
//                        }
//                    }
                    for (auto &segment : annotations.back().getSegments())
                    {
                        if ( auto segmentLabel = labels.at( segment.getLabel()); segmentLabel == l )
                            segments.push_back( segment.getSubsequence());
                    }
                }
            }
            return MCModel::train( trainingSegments, modelTrainer );
        }


        static BackboneProfiles
        _learnByBinarySegmentation( const BackboneProfiles &backbones,
                                    const BackboneProfiles &backgrounds,
                                    const std::map<std::string_view, std::vector<std::string>> &trainingSequences,
                                    const ModelGenerator <Grouping> &modelTrainer )
        {
            std::map<std::string_view, std::vector<std::string_view >> trainingSegments;
            for (auto &[l, sequences] : trainingSequences)
            {
                auto &segments = trainingSegments[l];
                auto &backbone = backbones.at( l );
                auto &background = backgrounds.at( l );

                for (auto &sequence : sequences)
                {
                    SequenceAnnotator annotator( sequence, _extractBinaryScores( sequence, backbone, background ));
                    std::vector<SequenceAnnotation> annotations = annotator.annotate( MAX_SEGMENTS );

                    for (auto &segment : annotations.back().getSegments())
                    {
                        if ( segment.getLabel() == 0 )
                            segments.push_back( segment.getSubsequence());
                    }

                }
            }
            return MCModel::train( trainingSegments, modelTrainer );
        }

        ScoredLabels _predict( std::string_view sequence ) const override
        {
            const SequenceAnnotator annotator( sequence, _scoringFunctions );
            const std::vector<SequenceAnnotation> annotations = annotator.annotate( MAX_SEGMENTS );

            std::map<std::string_view, double> propensities;

//            for( auto &annotation : annotations )
//            {
//                for (auto &segment : annotation.getSegments())
//                {
//                    for( auto &[learnerLabel,learner] : _segmentationLearners )
//                    {
//                        propensities[learnerLabel] += learner->propensity( segment.getSubsequence());
//                    }
//                }
//            }

            for (auto &segment : annotations.back().getSegments())
            {
                for (auto &[learnerLabel, learner] : _segmentationLearners)
                {
                    propensities[learnerLabel] += learner->propensity( segment.getSubsequence());
                }
            }

            propensities = minmaxNormalize( std::move( propensities ));

            ScoredLabels matchSet( _segmentationLearners.size());
            for (auto &[label, relativeAffinity] : propensities)
                matchSet.emplace( label, relativeAffinity );

            return matchSet;
        }

        ScoredLabels _predict2( std::string_view sequence ) const
        {
            std::map<std::string_view, double> propensities;

            for (auto&[label, backbone] :_segmentationLearners)
            {
                propensities[label] = backbone->propensity( sequence );
            }

            propensities = minmaxNormalize( std::move( propensities ));

            ScoredLabels matchSet( _segmentationLearners.size());
            for (auto &[label, relativeAffinity] : propensities)
                matchSet.emplace( label, relativeAffinity );

            return matchSet;
        }

    protected:
        const BackboneProfiles &_backbones;
        const BackboneProfiles &_background;
        const std::vector< ScoreFunction > &_scoringFunctions;
        const ModelGenerator <Grouping> _modelTrainer;
        BackboneProfiles _segmentationLearners;
    };
}

#endif //MARKOVIAN_FEATURES_MCSEGMENTATIONCLASSIFIER_H
