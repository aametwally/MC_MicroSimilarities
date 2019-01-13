//
// Created by asem on 13/08/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP

#include "AbstractMC.hpp"
#include "MCFeatures.hpp"
#include "SVMModel.hpp"
#include "MLConfusedMC.hpp"
#include "SimilarityMetrics.hpp"

namespace MC {

    template<size_t States>
    class SVMMCParameters : private SVMModel, public MLConfusedMC
    {
    public:
        using MCModel = AbstractMC<States>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<States>;
        using HeteroHistograms = typename MCModel::TransitionMatrices2D;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using BackboneProfile = typename MCModel::BackboneProfile;
        using ModelTrainer = ModelGenerator<States>;
        using SimilarityFunction = MetricFunction<Histogram>;

    public:
        explicit SVMMCParameters( const BackboneProfiles &backbones,
                                  const BackboneProfiles &background,
                                  const std::map<std::string_view, std::vector<std::string >> &training,
                                  ModelTrainer trainer,
                                  SimilarityFunction similarity,
                                  std::optional<double> lambda = 1,
                                  std::optional<double> gamma = 10 )
                : _modelTrainer( trainer ), _similarity( similarity ) ,
                  SVMModel( lambda, gamma )
        {
            fit( backbones, background , training );
        }

        explicit SVMMCParameters( ModelTrainer trainer,
                                  SimilarityFunction similarity,
                                  std::optional<double> lambda = 1,
                                  std::optional<double> gamma = 10 )
                : _modelTrainer( trainer ), _similarity( similarity ) ,
                  SVMModel( lambda, gamma )
        {}

        virtual ~SVMMCParameters() = default;

        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string_view, std::vector<std::string >> &training )
        {
            _backbones = backbones;
            _background = background;
            _selection.emplace( std::move( AbstractMC<States>::populationFeatureSpace( _backbones->get())));
            MLConfusedMC::enableLDA( );
            MLConfusedMC::fit( training );
        }

        using AbstractClassifier::predict;
    protected:

        bool _validTraining() const override
        {
            return _backbones && _background && _selection &&
                   _backbones->get().size() == _background->get().size();
        }

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

        void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            SVMModel::fit( labels, std::move( f ));
        }

        ScoredLabels _predictML( const FeatureVector &f ) const override
        {
            return SVMModel::predict( f );
        }

    protected:
        const ModelTrainer _modelTrainer;
        const SimilarityFunction _similarity;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;
        std::optional<Selection> _selection;
    };
}

#endif //MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP
