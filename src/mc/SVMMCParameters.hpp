//
// Created by asem on 13/08/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP

#include "AbstractMC.hpp"
#include "MCFeatures.hpp"
#include "SVMModel.hpp"
#include "MLConfusedMC.hpp"

namespace MC {

    template<typename Grouping>
    class SVMMCParameters : protected SVMModel, protected MLConfusedMC
    {
    public:
        using MCModel = AbstractMC<Grouping>;
        using Histogram = typename MCModel::Histogram;
        using MCF = MCFeatures<Grouping>;
        using HeteroHistograms = typename MCModel::HeteroHistograms;
        using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
        using BackboneProfiles = typename MCModel::BackboneProfiles;
        using ModelTrainer = ModelGenerator<Grouping>;

    public:
        explicit SVMMCParameters( ModelTrainer trainer ,  double lambda = 0.001 , double gamma  = 0.5  )
                : _modelTrainer( trainer ) , SVMModel( lambda , gamma )
        {}


        void fit( const BackboneProfiles &backbones,
                  const BackboneProfiles &background,
                  const std::map<std::string_view, std::vector<std::string >> &training )
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
        std::optional<FeatureVector> _extractFeatures( std::string_view sequence ) const override
        {
            if ( auto model = _modelTrainer( sequence, _selectedKernels ); *model )
                return model->extractFlatFeatureVector( _selectedKernels );
            else return std::nullopt;
        }

        void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
        {
            SVMModel::fit( labels, std::move( f ));
        }

        std::string_view _predictML( const FeatureVector &f ) const override
        {
            return SVMModel::predict( f );
        }

        void _featureSelection( const std::map<std::string_view, std::vector<std::string >> &training )
        {
            _selectedKernels = MCModel::withinJointAllUnionKernels( training, _modelTrainer );
        }


    protected:
        const ModelTrainer _modelTrainer;

        Selection _selectedKernels;
//    std::vector<bool> _selectedFeaturesMask;

    };

}
#endif //MARKOVIAN_FEATURES_SVMMARKOVIANMODEL_HPP
