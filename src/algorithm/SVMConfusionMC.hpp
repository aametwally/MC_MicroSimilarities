//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_SVMCONFUSIONMC_HPP
#define MARKOVIAN_FEATURES_SVMCONFUSIONMC_HPP

#include "SVMMarkovianModel.hpp"
#include "MLConfusedMC.hpp"

namespace MC {

    template<typename Grouping>
    class SVMConfusionMC : protected SVMModel, protected MLConfusedMC
    {
        using Ops = MCOps<Grouping>;
        using MCModel = AbstractMC<Grouping>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<Grouping>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using ModelTrainer = typename MCModel::ModelTrainer;
        using HistogramsTrainer = typename MCModel::HistogramsTrainer;

    public:

        explicit SVMConfusionMC( ModelTrainer modelTrainer, HistogramsTrainer trainer )
                : _modelTrainer( modelTrainer )
        {}

        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string, std::vector<std::string >> &training )
        {
            _backbones = backbones;
            _background = background;
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
        std::optional<FeatureVector> extractFeatures( const std::string &sequence ) const override
        {
            FeatureVector f;
            for (auto &[cluster, backbone] : _backbones->get())
            {
                auto &bg = _background->get().at( cluster );
                f.push_back( backbone->propensity( sequence ) - bg->propensity( sequence ));
            }
            return f;
        }

        void fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            SVMModel::fit( labels , std::move( f ));
        }

        std::string_view predictML( const FeatureVector &f ) const override
        {
            return SVMModel::predict( f );
        }


    protected:
        std::optional<std::reference_wrapper<const BackboneProfiles >> _backbones;
        std::optional<std::reference_wrapper<const BackboneProfiles >> _background;

        ModelTrainer _modelTrainer;
    };

}
#endif //MARKOVIAN_FEATURES_SVMCONFUSIONMC_HPP
