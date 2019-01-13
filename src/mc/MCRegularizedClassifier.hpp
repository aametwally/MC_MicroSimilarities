//
// Created by asem on 07/01/19.
//

#ifndef MARKOVIAN_FEATURES_MCREGULARIZEDCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MCREGULARIZEDCLASSIFIER_HPP

#include "AbstractMC.hpp"
#include "AbstractClassifier.hpp"
#include "KNNMCParameters.hpp"
#include "RegularizedMC.hpp"

namespace MC
{

template < typename CoreMCModel , size_t States >
class MCRegularizedClassifier : public AbstractClassifier
{
    static_assert( CoreMCModel::t_States == States , "States mismatch!" );

    using Model = RegularizedMC<CoreMCModel , States>;
    using Histogram = typename Model::Histogram;
    using BackboneProfile = typename Model::BackboneProfile;
    using BackboneProfiles = typename Model::BackboneProfiles;
    using Similarity = MetricFunction<Histogram>;
    using LogOddsFunction = std::function<double( std::string_view )>;

    struct KNNClassifier : public KNNMCParameters<States , Model>
    {
        using Base = KNNMCParameters<States , Model>;
        using FeatureVector = typename Base::FeatureVector;
    public:
        using Base::Base;
    protected:
    };

    using MG = ModelGenerator<States , Model>;

public:
    explicit MCRegularizedClassifier( Order order , const Similarity similarity )
            : _order( order ) , _similarity( similarity ) ,
              _knn( Order( 3 ) , MG::template create<Model>( order ) , _similarity ) {}

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
        _logOddsFunction = _extractScoringFunctions( _regularizedBackbones , _regularizedBackgrounds );
        _knn.fit( _regularizedBackbones , _regularizedBackgrounds , trainingClusters );
    }

protected:
    bool _validTraining() const override
    {
        return _regularizedBackbones.size() == _regularizedBackgrounds.size()
               && !_regularizedBackgrounds.empty();
    }

    ScoredLabels _predict1( std::string_view sequence ) const
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

    ScoredLabels _predict2( std::string_view sequence ) const
    {
        std::map<std::string_view , double> voter;
        const size_t k = _regularizedBackbones.size();
//        Model m( _order );
//        m.addSequence( sequence );
//        m.regularize();
        CoreMCModel m( _order );
        m.train( sequence );
        for ( const auto &[order , isoHistograms] : m.histograms().get())
        {
            for ( const auto &[id , queryHistogram] : isoHistograms )
            {
                ScoredLabels pq( k );
                for ( const auto &[clusterName , profile] : _regularizedBackbones )
                {
                    const auto &backboneHistograms = profile->histograms().get();
                    const auto &backgroundHistograms =
                            _regularizedBackgrounds.at( clusterName )->histograms().get();

                    auto backboneHistogram = backboneHistograms( order , id );
                    auto backgroundHistogram = backgroundHistograms( order , id );

                    if ( backboneHistogram && backgroundHistogram )
                    {
                        auto val = _similarity( queryHistogram - backgroundHistogram->get() ,
                                                backboneHistogram->get() - backgroundHistogram->get());
                        pq.emplace( clusterName , val );
                    }
                }
                pq.forTopK( 1 , [&]( const auto &candidate , size_t index )
                {
                    std::string_view label = candidate.label();
                    double val = (1) / (index + 1);
                    voter[label] += val;
                } );
            }
        }

        voter = minmaxNormalize( std::move( voter ));

        ScoredLabels scoredQueue( k );
        for ( auto[label , votes] : voter )
            scoredQueue.emplace( label , votes );
        for ( auto &[label , _] : _regularizedBackbones )
        {
            scoredQueue.findOrInsert( label );
        }
        return scoredQueue;
    }

    ScoredLabels _predict3( std::string_view sequence ) const
    {
        std::map<std::string_view , double> propensitites;
        std::vector<std::pair<std::string_view , double >> scores( sequence.length() ,
                                                                   std::make_pair( _logOddsFunction.cbegin()->first ,
                                                                                   -inf ));
        for ( auto i = 0; i < sequence.length(); ++i )
        {
            auto subsequence = sequence.substr( 0 , i + 1 );
            for ( auto&[label , fn] : _logOddsFunction )
            {
                double score = fn( subsequence );
                if ( score > scores.at( i ).second )
                {
                    scores.at( i ) = std::make_pair( label , score );
                }
            }
            if ( scores.at( i ).second > 0 )
            {
                propensitites[scores.at( i ).first] += scores.at( i ).second;
            }
        }

        propensitites = minmaxNormalize( std::move( propensitites ));

        ScoredLabels matchSet( _logOddsFunction.size());
        for ( auto &[label , relativeAffinity] : propensitites )
            matchSet.emplace( label , relativeAffinity );

        return matchSet;
    }


    ScoredLabels _predict4( std::string_view sequence ) const
    {
        return _knn.scoredPredictions( sequence );
    }

    ScoredLabels _predict( std::string_view sequence ) const override
    {
        return _predict4( sequence );
    }

    static std::map<std::string_view , LogOddsFunction>
    _extractScoringFunctions( const BackboneProfiles &profiles , const BackboneProfiles &backgrounds )
    {
        std::map<std::string_view , LogOddsFunction> scoringFunctions;
        for ( auto &[l , profile] : profiles )
        {
            auto &background = backgrounds.at( l );
            scoringFunctions.emplace( l , [&]( std::string_view query ) -> double
            {
                assert( !query.empty());
                char state = query.back();
                query.remove_suffix( 1 );
                return profile->transitionalPropensity( query , state ) -
                       background->transitionalPropensity( query , state );
            } );
        }
        return scoringFunctions;
    }


protected:
    const Order _order;
    const Similarity _similarity;
    std::map<std::string_view , LogOddsFunction> _logOddsFunction;

    BackboneProfiles _regularizedBackbones;
    BackboneProfiles _regularizedBackgrounds;
    KNNClassifier _knn;
};

}
#endif //MARKOVIAN_FEATURES_MCREGULARIZEDCLASSIFIER_HPP
