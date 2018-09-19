//
// Created by asem on 13/09/18.
//

#ifndef MARKOVIAN_FEATURES_KNNMCPARAMETERS_HPP
#define MARKOVIAN_FEATURES_KNNMCPARAMETERS_HPP

#include <dlib/statistics/dpca.h>
#include "AbstractMC.hpp"
#include "MCFeatures.hpp"
#include "KNNModel.hpp"
#include "MLConfusedMC.hpp"
#include "SimilarityMetrics.hpp"


namespace MC {
    template<typename Grouping>
    class KNNMCParameters : protected KNNModel<Euclidean>, protected MLConfusedMC
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
        explicit KNNMCParameters( ModelTrainer trainer, SimilarityFunction similarity )
                : _modelTrainer( trainer ), _similarity( similarity )
        {}


        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string_view, std::vector<std::string >> &training )
        {
            _backbones = backbones;
            _background = background;
            _centroid = MCModel::backgroundProfile( training, _modelTrainer, std::nullopt );
//            _featureSelection( training );
            MLConfusedMC::fit( training );
        }


        virtual std::vector<std::string_view> predict( const std::vector<std::string> &test ) const
        {
            std::vector<std::string_view> labels;
            for (auto &seq : test)
                labels.emplace_back( MLConfusedMC::predict( seq ));

            return labels;
        }

    protected:
        std::optional<FeatureVector> extractFeatures( const std::string_view sequence ) const override
        {
            if ( _backbones && _background )
            {
                if ( auto sample = _modelTrainer( sequence ); *sample )
                {
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
                                auto centroidHistogramOpt = _centroid->histogram( order , id );
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
                } else return std::nullopt;
            } else throw std::runtime_error( "Bad training!" );

        }

        void fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            KNNModel::fit( labels, std::move( f ));
        }

        std::string_view predictML( const FeatureVector &f ) const override
        {
            return KNNModel::predict( f );
        }

        void _featureSelection( const std::map<std::string, std::vector<std::string >> &training )
        {

        }


    protected:
        const ModelTrainer _modelTrainer;
        const SimilarityFunction _similarity;
        BackboneProfile _centroid;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;

//    std::vector<bool> _selectedFeaturesMask;

    };

}


#endif //MARKOVIAN_FEATURES_KNNMCPARAMETERS_HPP
