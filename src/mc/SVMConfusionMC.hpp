//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_SVMCONFUSIONMC_HPP
#define MARKOVIAN_FEATURES_SVMCONFUSIONMC_HPP

#include "SimilarityMetrics.hpp"
#include "SVMMCParameters.hpp"
#include "MLConfusedMC.hpp"
#include "AbstractClassifier.hpp"
#include "MCPropensityClassifier.hpp"
#include "MacroSimilarityClassifier.hpp"
#include "MicroSimilarityVotingClassifier.hpp"
#include "MCKmersClassifier.hpp"

namespace MC {

    template<typename Grouping>
    class SVMConfusionMC : protected SVMModel, protected MLConfusedMC
    {
        using MCModel = AbstractMC<Grouping>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<Grouping>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using ModelTrainer = ModelGenerator<Grouping>;
        using Similarity = MetricFunction<Histogram>;

    public:

        explicit SVMConfusionMC( ModelTrainer modelTrainer,
                                 std::optional<double> lambda = 1,
                                 std::optional<double> gamma = 10 )
                : _modelTrainer( modelTrainer ), SVMModel( lambda, gamma )
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
            _ensemble.emplace( ClassificationEnum::Propensity,
                               new MCPropensityClassifier<Grouping>( backbones, background ));

            _ensemble.emplace( ClassificationEnum::Accumulative,
                               new MacroSimilarityClassifier<Grouping>( backbones, background,
                                                                        selection, trainer, similarity ));

            _ensemble.emplace( ClassificationEnum::Voting,
                               new MicroSimilarityVotingClassifier<Grouping>( backbones, background,
                                                                              selection, trainer, similarity ));

            _ensemble.emplace( ClassificationEnum::KMERS,
                               new MCKmersClassifier<Grouping>( backbones, background ));
            MLConfusedMC::setLDA( backbones.size() );
            MLConfusedMC::fit( training );
        }


        std::vector<std::string_view> predict( const std::vector<std::string> &test ) const
        {
            if ( _backbones && _background )
            {
                std::vector<std::string_view> labels;
                for (auto &seq : test)
                    labels.emplace_back( MLConfusedMC::predict( seq ));

                return labels;
            } else throw std::runtime_error( fmt::format( "Bad training" ));
        }

    protected:
        std::optional<FeatureVector> _extractFeatures( std::string_view sequence ) const override
        {
            FeatureVector f;
            for (auto &[enumm, classifier] : _ensemble)
            {
                auto relativeAffinities = classifier->scoredPredictions( sequence );
                for (auto &[cluster, _] : _backbones->get())
                    f.push_back( relativeAffinities.at( cluster ));
            }
            return f;
        }

        void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            SVMModel::fit( labels, std::move( f ));
        }

        std::string_view _predictML( const FeatureVector &f ) const override
        {
            return SVMModel::predict( f );
        }


    protected:
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;
        std::map<ClassificationEnum, std::unique_ptr<AbstractClassifier >> _ensemble;

        ModelTrainer _modelTrainer;
    };

}
#endif //MARKOVIAN_FEATURES_SVMCONFUSIONMC_HPP
