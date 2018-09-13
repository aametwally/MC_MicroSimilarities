//
// Created by asem on 13/08/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP

#include "AbstractMC.hpp"
#include "MCOperations.hpp"
#include "MCFeatures.hpp"
#include "SVMModel.hpp"
#include "MLConfusedMC.hpp"

namespace MC {

    template<typename Grouping>
    class SVMMarkovianModel : protected SVMModel, protected MLConfusedMC
    {
    public:
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
        explicit SVMMarkovianModel( HistogramsTrainer trainer )
                : _histogramsTrainer( trainer )
        {}


        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string, std::vector<std::string >> &training )
        {
            _featureSelection( training );
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
        std::optional<FeatureVector> extractFeatures( const std::string &sequence ) const override
        {
            if ( auto histograms = _histogramsTrainer( {sequence}, _selectedKernels ); histograms )
            {
                auto flatFeatures = Ops::extractFlatFeatureVector( histograms.value(), _selectedKernels );
                return flatFeatures;
            } else return std::nullopt;
        }

        void fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            SVMModel::fit( labels, std::move( f ));
        }

        std::string_view predictML( const FeatureVector &f ) const override
        {
            return SVMModel::predict( f );
        }

        void _featureSelection( const std::map<std::string, std::vector<std::string >> &training )
        {
            _selectedKernels = Ops::withinJointAllUnionKernels( training, _histogramsTrainer );
        }


    protected:
        const HistogramsTrainer _histogramsTrainer;

        Selection _selectedKernels;
//    std::vector<bool> _selectedFeaturesMask;

    };

}
#endif //MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP
