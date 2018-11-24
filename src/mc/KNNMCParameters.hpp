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


namespace MC {
    template<typename Grouping>
    class KNNMCParameters : private KNNModel<Euclidean>, public MLConfusedMC
    {
    public:
        using MCModel = AbstractMC<Grouping>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<Grouping>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using BackboneProfile = typename MCModel::BackboneProfile;
        using ModelTrainer = ModelGenerator<Grouping>;

        using SimilarityFunction = MetricFunction<Histogram>;

    public:
        explicit KNNMCParameters( size_t k , ModelTrainer trainer, SimilarityFunction similarity )
                : _modelTrainer( trainer ), _similarity( similarity ), KNNModel( k )
        {}

        explicit KNNMCParameters( const BackboneProfiles &backbones,
                                  const BackboneProfiles &background,
                                  const std::map<std::string_view, std::vector<std::string >> &training,
                                  size_t k , ModelTrainer trainer, SimilarityFunction similarity )
                : _modelTrainer( trainer ), _similarity( similarity ), KNNModel( k )
        {
            fit( backbones , background , training );
        }

        virtual ~KNNMCParameters() = default;

        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string_view, std::vector<std::string >> &training )
        {
            _backbones = backbones;
            _background = background;
            _centroid = MCModel::backgroundProfile( training, _modelTrainer, std::nullopt );
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
                std::map<std::string_view, std::vector<double >> similarities;
                std::map<std::string_view, std::vector<double >> backgroundSimilarities;
                static const auto noMeasurement = std::vector<double>( _backbones->get().size(), 0.0 );
                for (auto &[label, profile] : _backbones->get())
                {
                    auto &classSimilarities = similarities[label];
                    auto &bgSimilarities = backgroundSimilarities[label];
                    auto &bgHistograms = _background->get().at( label );
                    for (auto &[order, isoClassHistograms] : profile->histograms().get())
                    {
                        for (auto &[id, classHistogram] : isoClassHistograms)
                        {
                            auto bgHistogramOpt = bgHistograms->histogram( order, id );
                            auto sampleHistogramOpt = sample->histogram( order, id );
                            auto centroidHistogramOpt = _centroid->histogram( order, id );
                            if ( sampleHistogramOpt && bgHistogramOpt && centroidHistogramOpt )
                            {
                                auto &histogram = sampleHistogramOpt->get();
                                auto &bgHistogram = bgHistogramOpt->get();
                                auto &centroidHistogram = centroidHistogramOpt->get();

                                classSimilarities.push_back(
                                        _similarity( histogram - centroidHistogram,
                                                     classHistogram - centroidHistogram ));
                                bgSimilarities.push_back(
                                        _similarity( histogram - centroidHistogram,
                                                     bgHistogram - centroidHistogram ));
                            } else
                            {
                                classSimilarities.push_back( 0 );
                                bgSimilarities.push_back( 0 );
                            }
                        }
                    }
                }

                std::vector<double> flatFeatures;
                for (auto &[label, sim] :  similarities)
                {
                    auto &bgSim = backgroundSimilarities.at( label );
                    for (size_t i = 0; i < sim.size(); ++i)
                    {
                        flatFeatures.push_back( sim[i] - bgSim[i] );
                    }
                }
                return flatFeatures;
            } else throw std::runtime_error( "Bad training!" );

        }

        void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            KNNModel::fit( labels, std::move( f ));
        }

        ScoredLabels _predictML( const FeatureVector &f ) const override
        {
            return KNNModel::predict( f );
        }

        void _featureSelection( const std::map<std::string, std::vector<std::string >> &training )
        {

        }


    protected:
        bool _validTraining() const override
        {
            return _backbones && _background && _centroid
                   && _backbones->get().size() == _background->get().size();
        }

    private:
        BackboneProfile _centroid;

        const ModelTrainer _modelTrainer;
        const SimilarityFunction _similarity;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;

//    std::vector<bool> _selectedFeaturesMask;

    };

}


#endif //MARKOVIAN_FEATURES_KNNMCPARAMETERS_HPP
