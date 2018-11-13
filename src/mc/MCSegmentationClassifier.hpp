//
// Created by asem on 11/11/18.
//

#ifndef MARKOVIAN_FEATURES_MCSEGMENTATIONCLASSIFIER_H
#define MARKOVIAN_FEATURES_MCSEGMENTATIONCLASSIFIER_H

#include "AbstractClassifier.hpp"
#include "AbstractMC.hpp"
#include "SequenceAnnotator.hpp"

namespace MC
{
template < typename Grouping >
class MCSegmentationClassifier : public AbstractClassifier
{
    using MCModel = AbstractMC<Grouping>;

    using BackboneProfiles = typename MCModel::BackboneProfiles;

public:
    explicit MCSegmentationClassifier( const BackboneProfiles &backbones ,
                                       const BackboneProfiles &background ,
                                       const std::map<std::string_view , std::vector<std::string>> &trainingSequences ,
                                       const ModelGenerator<Grouping> &modelTrainer )
            : _backbones( backbones ) ,
              _background( background ) ,
              _segmentationLearners( _learnBySegmentation( backbones , trainingSequences , modelTrainer )),
              _modelTrainer( modelTrainer )
    {
    }


protected:
    bool _validTraining() const override
    {
        return _backbones.size() == _background.size();
    }

    static std::vector<std::vector<double >> _extractScores( std::string_view query , const BackboneProfiles &profiles )
    {
        std::vector<std::vector<double >> scoresForward;

        for ( auto &[l , profile] : profiles )
        {
            auto forward = profile->compensatedPropensityVector( query );
//            auto forwardBackground = _background.at(l)->forwardPropensityVector( query );
//            std::transform(forward.cbegin(), forward.cend(), forwardBackground.cbegin(),
//                           forward.begin(), std::minus<double>());
            scoresForward.emplace_back( std::move( forward ));
        }

        return scoresForward;
    }

    static BackboneProfiles _learnBySegmentation( const BackboneProfiles &backbones ,
                                                  const std::map<std::string_view , std::vector<std::string>> &trainingSequences ,
                                                  const ModelGenerator<Grouping> &modelTrainer )
    {
        std::vector<std::string_view> labels;
        std::transform( backbones.cbegin() , backbones.cend() , std::back_inserter( labels ) ,
                        []( const auto &p ) { return p.first; } );

        std::map< std::string_view , std::vector< std::string_view >> trainingSegments;
        for ( auto &[l , sequences] : trainingSequences )
        {
            auto &segments = trainingSegments[l];
            for ( auto &sequence : sequences )
            {
                SequenceAnnotator annotator( sequence , _extractScores( sequence , backbones ));
                std::vector<SequenceAnnotation> annotations = annotator.annotate( 5 );
                for ( auto &annotation : annotations )
                {
                    for ( auto &segment : annotation.getSegments())
                    {
                        if( auto segmentLabel = labels.at( segment.getLabel()); segmentLabel == l )
                            segments.push_back( segment.getSubsequence());
                    }
                }
            }
        }
        return MCModel::train( trainingSegments, modelTrainer );
    }

    ScoredLabels _individualAAIdentitiesPredictor( const SequenceAnnotation &annotation ) const
    {
        assert( annotation.getSegments().size() == 1 );
        std::map<std::string_view , double> relativeIdentities;

        auto it = _backbones.cbegin();
        for ( auto &[l , count] : annotation.getSegments().front().getCounter())
        {
            relativeIdentities.emplace( it->first , count );
            ++it;
        }

        relativeIdentities = minmaxNormalize( std::move( relativeIdentities ));
        ScoredLabels matchSet( annotation.getSegments().size());
        for ( auto &[label , identity] : relativeIdentities )
            matchSet.emplace( label , identity );

        return matchSet;
    }

    ScoredLabels _predict( std::string_view sequence ) const override
    {
        std::map<std::string_view, double> propensitites;

        for (auto&[label, backbone] :_segmentationLearners)
        {
            propensitites[label] = backbone->propensity( sequence );
        }

        propensitites = minmaxNormalize( std::move( propensitites ));

        ScoredLabels matchSet( _segmentationLearners.size() );
        for (auto &[label, relativeAffinity ] : propensitites)
            matchSet.emplace( label, relativeAffinity );

        return matchSet;
    }

protected:
    const BackboneProfiles &_backbones;
    const BackboneProfiles &_background;

    const ModelGenerator<Grouping> _modelTrainer;
    BackboneProfiles _segmentationLearners;
};
}

#endif //MARKOVIAN_FEATURES_MCSEGMENTATIONCLASSIFIER_H
