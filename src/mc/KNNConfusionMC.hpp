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
#include "MCKmersClassifier.hpp"
#include "KNNMCParameters.hpp"
#include "SVMMCParameters.hpp"

namespace MC {
    template<size_t States>
    class KNNConfusionMC : protected KNNModel<Euclidean>, public MLConfusedMC
    {
        using KNN = KNNModel<Euclidean>;
        using MCModel = AbstractMC<States>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<States>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using ModelTrainer =  ModelGenerator<States>;
        using Similarity = MetricFunction<Histogram>;

    public:
        explicit KNNConfusionMC( size_t k )
                : KNN( k )
        {}

        virtual ~KNNConfusionMC() = default;

        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string_view, std::vector<std::string >> &training,
                  ModelTrainer trainer,
                  Similarity similarity,
                  std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
        {
            _backbones = backbones;
            _background = background;
            _ensemble.emplace( ClassificationEnum::Propensity,
                               new MCPropensityClassifier<States>( backbones, background ));

            _ensemble.emplace( ClassificationEnum::Accumulative,
                               new MacroSimilarityClassifier<States>( backbones, background,
                                                                        trainer, similarity, selection ));

            _ensemble.emplace( ClassificationEnum::Voting,
                               new MicroSimilarityVotingClassifier<States>( backbones, background,
                                                                              trainer, similarity, selection ));

//            _ensemble.emplace( ClassificationEnum::KNN,
//                               new KNNMCParameters<Grouping>( backbones, background, training,
//                                                              7, trainer, similarity ));
//
//            _ensemble.emplace( ClassificationEnum::SVM,
//                               new SVMMCParameters<Grouping>( backbones, background, training, trainer, similarity ));

//            _ensemble.emplace( ClassificationEnum::KMERS,
//                               new MCKmersClassifier<Grouping>( backbones, background ));
            MLConfusedMC::enableLDA();
            MLConfusedMC::fit( training );
        }

        using AbstractClassifier::predict;
    protected:
        bool _validTraining() const override
        {
            return _backbones && _background && !_ensemble.empty();
        }

        FeatureVector _extractFeatures( std::string_view sequence ) const override
        {
            FeatureVector f;
            for (auto &[enumm, classifier] : _ensemble)
            {
                auto propensityPredictions = classifier->scoredPredictions( sequence ).toMap();
                for (auto &[cluster, _] : _backbones->get())
                    f.push_back( propensityPredictions.at( cluster ));
            }
            f.push_back( sequence.length());
            return f;
        }

        void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            KNN::fit( labels, std::move( f ));
        }

        ScoredLabels _predictML( const FeatureVector &f ) const override
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
