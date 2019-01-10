//
// Created by asem on 07/01/19.
//

#ifndef MARKOVIAN_FEATURES_MCREGULARIZEDCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MCREGULARIZEDCLASSIFIER_HPP

#include "AbstractMC.hpp"
#include "AbstractClassifier.hpp"

#include "RegularizedMC.hpp"

namespace MC
{

template < size_t States >
class MCRegularizedClassifier : public AbstractClassifier
{
    using Model = RegularizedMC<States>;
    using BackboneProfile = std::unique_ptr<Model>;
    using BackboneProfiles = std::map<std::string_view , std::unique_ptr<Model>>;

public:
    explicit MCRegularizedClassifier( Order order )
            : _order( order )
    {}

    virtual ~MCRegularizedClassifier() = default;

    void runTraining( const std::map<std::string_view , std::vector<std::string >> &trainingClusters )
    {
        for ( const auto &[label , sequences] : trainingClusters )
        {
            auto backboneIt = _regularizedBackbones.emplace( label , std::make_unique<Model>( _order )).first;
            auto &backbone = backboneIt->second;
            backbone->addSequences( sequences );

            auto backgroundIt = _regularizedBackgrounds.emplace( label , std::make_unique<Model>( _order )).first;
            auto &background = backgroundIt->second;
            for ( const auto &[backgroundLabel , backgroundSequences] : trainingClusters )
            {
                if ( backgroundLabel != label )
                {
                    background->addSequences( backgroundSequences );
                }
            }

            backbone->regularize();
            background->regularize();
        }
    }

protected:
    bool _validTraining() const override
    {
        return _regularizedBackbones.size() == _regularizedBackgrounds.size()
               && !_regularizedBackgrounds.empty();
    }

    ScoredLabels _predict( std::string_view sequence ) const override
    {
        std::map<std::string_view , double> propensitites;

        for ( auto&[label , backbone] : _regularizedBackbones )
        {
            auto &bg = _regularizedBackgrounds.at( label );
            double logOdd = backbone->propensity( sequence ) - bg->propensity( sequence );
            propensitites[label] = logOdd;
        }


        propensitites = minmaxNormalize( std::move( propensitites ));

        ScoredLabels matchSet( _regularizedBackbones.size());
        for ( auto &[label , relativeAffinity] : propensitites )
            matchSet.emplace( label , relativeAffinity );

        return matchSet;
    }

protected:
    const Order _order;
    BackboneProfiles _regularizedBackbones;
    BackboneProfiles _regularizedBackgrounds;
};

}
#endif //MARKOVIAN_FEATURES_MCREGULARIZEDCLASSIFIER_HPP
