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
        using ScoringFunction = std::function<double(std::string_view)>;

    public:
        explicit MCPropensityClassifier( const BackboneProfiles &backbones,
                                         const BackboneProfiles &background )
                : _backbones( backbones ),
                  _background( background ),
                  _scoringFunctions(_extractScoringFunctions( backbones , background ))
        {
        }


        virtual ~MCPropensityClassifier() = default;

    protected:
        bool _validTraining() const override
        {
            return _backbones.size() == _background.size();
        }

        static std::map< std::string_view , ScoringFunction>
        _extractScoringFunctions( const BackboneProfiles &backbones , const BackboneProfiles &backgrounds )
        {
            std::map< std::string_view , ScoringFunction> scoringFunctions;
            for (auto &[l, profile] : backbones)
            {
                auto &background = backgrounds.at( l );
                scoringFunctions.emplace( l, [&]( std::string_view query ) -> double {
                    assert( !query.empty());
                    char state = query.back();
                    query.remove_suffix( 1 );
                    return profile->transitionalPropensity( query, state ) -
                           background->transitionalPropensity( query, state );
                } );
            }
            return scoringFunctions;
        }


        ScoredLabels _predict( std::string_view sequence ) const override
        {
            std::map<std::string_view, double> propensitites;

            for( auto &[l,fn] : _scoringFunctions )
            {
                auto &propensity = propensitites[l];
                for( auto i = 1 ; i <= sequence.length() ; ++i )
                    propensity += fn( sequence.substr( 0 , i ) );
            }

//            for (auto&[label, backbone] :_backbones)
//            {
//                auto &bg = _background.at( label );
//                double logOdd = backbone->propensity( sequence ) - bg->propensity( sequence );
//                propensitites[label] = logOdd;
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
        std::map<std::string_view , ScoringFunction> _scoringFunctions;
    };
}


#endif //MARKOVIAN_FEATURES_MCPROPENSITYCLASSIFIER_HPP
