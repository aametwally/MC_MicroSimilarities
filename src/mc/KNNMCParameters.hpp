//
// Created by asem on 13/09/18.
//

#ifndef MARKOVIAN_FEATURES_KNNMCPARAMETERS_HPP
#define MARKOVIAN_FEATURES_KNNMCPARAMETERS_HPP

#include "AbstractMC.hpp"
#include "MCFeatures.hpp"
#include "KNNModel.hpp"
#include "MLConfusedMC.hpp"
#include "SimilarityMetrics.hpp"

namespace MC
{
template < size_t States , typename CoreModel = AbstractMC <States> >
class KNNMCParameters : private KNNModel<Euclidean> , public MLConfusedMC
{
public:
    using MCModel = CoreModel;
    using Histogram = typename MCModel::Histogram;
    using MCF = MCFeatures<States>;
    using TransitionMatrices2D = typename MCModel::TransitionMatrices2D;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using BackboneProfile = typename MCModel::BackboneProfile;
    using ModelTrainer = ModelGenerator<States , CoreModel>;

    using SimilarityFunction = MetricFunction<Histogram>;

public:
    explicit KNNMCParameters( size_t k , ModelTrainer trainer , SimilarityFunction similarity )
            : _modelTrainer( trainer ) , _similarity( similarity ) , KNNModel( k ) {}

    virtual ~KNNMCParameters() = default;

    void fit( const BackboneProfiles &backbones ,
              const BackboneProfiles &background ,
              const std::map<std::string_view , std::vector<std::string >> &training )
    {
        _backbones = backbones;
        _background = background;
        _selection.emplace( std::move( AbstractMC<States>::populationFeatureSpace( _backbones->get())));
        MLConfusedMC::enableLDA();
        MLConfusedMC::fit( training );
    }

    using AbstractClassifier::predict;
protected:
    FeatureVector _extractFeatures( std::string_view sequence ) const override
    {
        if ( _validTraining())
        {
            auto sample = _modelTrainer( sequence );
            std::map<std::string_view , std::vector<double >> similarities;
            for ( auto &[label , profile] : _backbones->get())
            {
                auto &classSimilarities = similarities[label];
                const auto &bbHistograms = profile->histograms().get();
                const auto &bgHistograms = _background->get().at( label )->histograms().get();
                const auto &sampleHistograms = sample->histograms().get();
                for ( auto &[order , isoIDs] : _selection.value())
                {
                    auto bbIsoHistograms = bbHistograms( order );
                    auto bgIsoHistograms = bgHistograms( order );
                    auto sampleIsoHistograms = sampleHistograms( order );
                    if ( bbIsoHistograms && bgIsoHistograms && sampleIsoHistograms )
                    {
                        for ( auto id : isoIDs )
                        {
                            auto bgHistogram = bgIsoHistograms.value()( id );
                            auto bbHistogram = bbIsoHistograms.value()( id );
                            auto sampleHistogram = sampleIsoHistograms.value()( id );

                            if ( sampleHistogram && bbHistogram && bgHistogram )
                            {
                                auto diff1 = sampleHistogram->get() - bgHistogram->get();
                                auto diff2 = bbHistogram->get() - bgHistogram->get();
                                double similarity = _similarity( diff1 , diff2 );
                                assert( !std::isnan( similarity ));
                                classSimilarities.push_back( similarity );
                            } else
                            {
                                classSimilarities.push_back( 0 );
                            }
                        }
                    }
                }
            }

            std::vector<double> flatFeatures;
            flatFeatures.reserve( similarities.size() * similarities.begin()->second.size());
            for ( auto &[label , sim] :  similarities )
            {
                flatFeatures.insert( flatFeatures.end() ,
                                     std::make_move_iterator( sim.begin()) ,
                                     std::make_move_iterator( sim.end()));
            }
            return flatFeatures;
        } else throw std::runtime_error( "Bad training!" );
    }

    void _fitML( const std::vector<std::string_view> &labels , std::vector<FeatureVector> &&f ) override
    {
        KNNModel::fit( labels , std::move( f ));
    }

    ScoredLabels _predictML( const FeatureVector &f ) const override
    {
        return KNNModel::predict( f );
    }

    void _featureSelection( const std::map<std::string , std::vector<std::string >> &training )
    {

    }

protected:
    bool _validTraining() const override
    {
        return _backbones && _background && _selection &&
               _backbones->get().size() == _background->get().size();
    }

protected:
    const ModelTrainer _modelTrainer;
    const SimilarityFunction _similarity;
    std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
    std::optional<std::reference_wrapper<const BackboneProfiles >> _background;
    std::optional<Selection> _selection;
};

}


#endif //MARKOVIAN_FEATURES_KNNMCPARAMETERS_HPP
