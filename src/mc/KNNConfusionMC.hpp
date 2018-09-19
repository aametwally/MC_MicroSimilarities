//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_KNNCONFUSIONMC_H
#define MARKOVIAN_FEATURES_KNNCONFUSIONMC_H

#include "AbstractMC.hpp"
#include "MCFeatures.hpp"

#include "KNNModel.hpp"
#include "MLConfusedMC.hpp"

#include "MCPropensityClassifier.hpp"
#include "MicroSimilarityVotingClassifier.hpp"
#include "MacroSimilarityClassifier.hpp"

namespace MC {
    template<typename Grouping>
    class KNNConfusionMC : protected KNNModel<Euclidean>, protected MLConfusedMC
    {
        using KNN = KNNModel<Euclidean>;
        using MCModel = AbstractMC<Grouping>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<Grouping>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using ModelTrainer =  ModelGenerator<Grouping>;
        using Similarity = MetricFunction<Histogram>;

    public:
        explicit KNNConfusionMC( size_t k )
                : KNN( k )
        {}

        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string_view, std::vector<std::string >> &training,
                  ModelTrainer trainer,
                  const Selection &selection,
                  Similarity similarity )
        {
            _backbones = backbones;
            _background = background;
//            _ensemble.emplace( ClassificationEnum::Propensity,
//                               new MCPropensityClassifier<Grouping>( backbones, background ));

            _ensemble.emplace( ClassificationEnum::Accumulative,
                               new MacroSimilarityClassifier<Grouping>( backbones, background,
                                                                        selection, trainer, similarity ));

            _ensemble.emplace( ClassificationEnum::Voting,
                               new MicroSimilarityVotingClassifier<Grouping>( backbones, background,
                                                                              selection, trainer, similarity ));
            MLConfusedMC::fit( training );
        }


        std::vector<std::string_view> predict( const std::vector<std::string> &test ) const
        {
            if ( _backbones && _background && !_ensemble.empty())
            {
                std::vector<std::string_view> labels;
                for (auto &seq : test)
                    labels.emplace_back( MLConfusedMC::predict( seq ));

                return labels;
            } else throw std::runtime_error( fmt::format( "Bad training" ));
        }

    protected:
        std::optional<FeatureVector> extractFeatures( std::string_view sequence ) const override
        {
            FeatureVector f;
            for (auto &[enumm, classifier] : _ensemble)
            {
                auto propensityPredictions = classifier->scoredPredictions( sequence );
                for (auto &[cluster, _] : _backbones->get())
                    f.push_back( propensityPredictions.at( cluster ));
            }
            return f;
        }

        void fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            KNN::fit( labels, std::move( f ));
        }

        std::string_view predictML( const FeatureVector &f ) const override
        {
            return KNN::predict( f );
        }

    protected:
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;
        std::map<ClassificationEnum, std::unique_ptr<AbstractClassifier >> _ensemble;
    };
}
#endif //MARKOVIAN_FEATURES_KNNCONFUSIONMC_H
