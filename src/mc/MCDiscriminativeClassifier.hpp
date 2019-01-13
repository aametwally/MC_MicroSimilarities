//
// Created by asem on 23/11/18.
//

#ifndef MARKOVIAN_FEATURES_MCDISCRIMINATIVECLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MCDISCRIMINATIVECLASSIFIER_HPP

#include "AbstractClassifier.hpp"
#include "AbstractMC.hpp"

namespace MC {
    template<size_t States>
    class MCDiscriminativeClassifier : public AbstractClassifier
    {
        using MCModel = AbstractMC<States>;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using LogOddsFunction = std::function<double( std::string_view )>;
    public:
        explicit MCDiscriminativeClassifier( const BackboneProfiles &backbones,
                                             const BackboneProfiles &background )
                : _backbones( backbones ),
                  _background( background ),
                  _logOddsFunction( _extractScoringFunctions( backbones, background ))
        {
        }

        virtual ~MCDiscriminativeClassifier() = default;

    protected:
        bool _validTraining() const override
        {
            return _backbones.size() == _background.size();
        }

        static std::map<std::string_view, LogOddsFunction>
        _extractScoringFunctions( const BackboneProfiles &profiles, const BackboneProfiles &backgrounds )
        {
            std::map<std::string_view, LogOddsFunction> scoringFunctions;
            for (auto &[l, profile] : profiles)
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

            std::vector< std::pair< std::string_view , double >> scores( sequence.length() ,
                    std::make_pair( _logOddsFunction.cbegin()->first , -inf ) );

            for (auto i = 0; i < sequence.length(); ++i)
            {
                auto subsequence =  sequence.substr( 0, i + 1 );
                for (auto&[label, fn] : _logOddsFunction)
                {
                    double score = fn( subsequence );
                    if( score > scores.at( i ).second )
                    {
                        scores.at( i ) = std::make_pair( label , score );
                    }
                }
                if( scores.at( i ).second > 0 )
                {
                    propensitites[ scores.at( i ).first ] += scores.at( i ).second ;
                }
            }

            propensitites = minmaxNormalize( std::move( propensitites ));

            ScoredLabels matchSet( _logOddsFunction.size());
            for (auto &[label, relativeAffinity] : propensitites)
                matchSet.emplace( label, relativeAffinity );

            return matchSet;
        }

    protected:
        const BackboneProfiles &_backbones;
        const BackboneProfiles &_background;
        const std::map<std::string_view, LogOddsFunction> _logOddsFunction;
    };
}

#endif //MARKOVIAN_FEATURES_MCDISCRIMINATIVECLASSIFIER_HPP
