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

    public:
        explicit MCPropensityClassifier( const BackboneProfiles &backbones,
                                         const BackboneProfiles &background )
                : _backbones( backbones ),
                  _background( background )
        {
        }


    protected:
        bool _validTraining() const override
        {
            return _backbones.size() == _background.size();
        }

        ScoredLabels _predict( std::string_view sequence ) const override
        {
            std::map<std::string_view, double> propensitites;

            for (auto&[label, backbone] :_backbones)
            {
                auto &bg = _background.at( label );
                double logOdd = backbone->propensity( sequence ) - bg->propensity( sequence );
                propensitites[label] = logOdd;
            }

//            const double minPropensity = std::min_element( propensitites.cbegin() , propensitites.cend() , []( auto &p1 , auto &p2 ){
//                return p1.second < p2.second;
//            })->second;
//
//            double sum = 0;
//            for( auto &[label,value] : propensitites )
//            {
//                value -= minPropensity;
//                sum += value;
//            }

            propensitites = minmaxNormalize( std::move( propensitites ));

            ScoredLabels matchSet( _backbones.size() );
            for (auto &[label, relativeAffinity ] : propensitites)
                matchSet.emplace( label, relativeAffinity );

            return matchSet;
        }

    protected:
        const BackboneProfiles &_backbones;
        const BackboneProfiles &_background;

    };
}


#endif //MARKOVIAN_FEATURES_MCPROPENSITYCLASSIFIER_HPP
