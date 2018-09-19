//
// Created by asem on 18/09/18.
//

#ifndef MARKOVIAN_FEATURES_MCPROPENSITYCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MCPROPENSITYCLASSIFIER_HPP

#include "AbstractClassifier.hpp"
#include "AbstractMC.hpp"

namespace MC {
    template<typename Grouping>
    class MCPropensityClassifier : public AbstractClassifier
    {
        using MCModel = AbstractMC<Grouping>;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using PriorityQueue = typename MatchSet<Score>::Queue<std::string_view>;

    public:
        explicit MCPropensityClassifier( const BackboneProfiles &backbones,
                                         const BackboneProfiles &background )
                : AbstractClassifier( backbones.size()),
                  _backbones( backbones ),
                  _background( background )
        {
        }


    protected:
        bool _validTraining() const override
        {
            return _backbones.size() == _background.size() && _backbones.size() == _nLabels;
        }

        PriorityQueue _predict( std::string_view sequence ) const override
        {
            std::map<std::string_view, double> propensitites;
            double sum = 0;
            for (auto&[label, backbone] :_backbones)
            {
                auto &bg = _background.at( label );
                double logOdd = backbone->propensity( sequence ) - bg->propensity( sequence );
                propensitites[label] = logOdd;
                sum += logOdd;
            }

            PriorityQueue matchSet( _nLabels );
            for (auto &[label, logOdd] : propensitites)
                matchSet.emplace( label, logOdd / sum );

            return matchSet;
        }

    protected:
        const BackboneProfiles &_backbones;
        const BackboneProfiles &_background;

    };
}


#endif //MARKOVIAN_FEATURES_MCPROPENSITYCLASSIFIER_HPP
