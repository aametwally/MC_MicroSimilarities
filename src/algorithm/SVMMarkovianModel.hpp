//
// Created by asem on 13/08/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMODEL_HPP

#include "common.hpp"
#include "AbstractMC.hpp"
#include "MCOperations.hpp"
#include "MCFeatures.hpp"

#include "dlib/svm_threaded.h"
#include "dlib_utilities.hpp"
#include "dlib/statistics/dpca.h"

using MC::Selection;

template<typename Grouping>
class SVMMarkovianModel
{
public:
    using MCOps = MC::MCOps<Grouping>;
    using MCModel = MC::AbstractMC<Grouping>;
    using Histogram = typename MCModel::Histogram;
    using MCF = MC::MCFeatures<Grouping>;
    using HeteroHistograms = typename MCModel::HeteroHistograms;
    using HeteroHistogramsFeatures = typename MCModel::HeteroHistogramsFeatures;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using ModelTrainer = typename MCModel::ModelTrainer;
    using HistogramsTrainer = typename MCModel::HistogramsTrainer;

    using SampleType = dlib::matrix<double, 0, 0>;

    using SVMBinaryKernel = dlib::radial_basis_kernel<SampleType>;
    using Label = int;

    using SVMRBFKernel = dlib::radial_basis_kernel<SampleType>;
    using SVMRBFTrainer = dlib::krr_trainer<SVMRBFKernel>;
    using SVMBinaryTrainer = dlib::krr_trainer<SVMBinaryKernel>;
    using SVMTrainer = dlib::one_vs_one_trainer<dlib::any_trainer<SampleType>, Label>;
    using DecisionFunction = SVMTrainer::trained_function_type;

public:
    explicit SVMMarkovianModel( HistogramsTrainer trainer )
            : histogramsTrainer_( trainer )
    {}


    void fit( const std::map<std::string, std::vector<std::string >> &training )
    {

        SVMTrainer trainer;
        SVMBinaryTrainer btrainer;
//        histogramTrainer.set_lambda(0.00001);
        btrainer.set_kernel( SVMBinaryKernel( 0.01 ));
//        histogramTrainer.set_max_num_sv(10);

        trainer.set_trainer( btrainer );
        trainer.set_num_threads( std::thread::hardware_concurrency());

        _registerLabels( training );
        _featureSelection( training );

        std::vector<int> labels;
        std::vector<SampleType> featuresVector;
        for (auto &[trainLabel, trainSeqs] : training)
        {
            int index = _label2Index.at( trainLabel );
            auto flatFeatures = MCOps::extractFlatFeatureVector(
                    histogramsTrainer_( trainSeqs , _selectedKernels ).value() , _selectedKernels );
            featuresVector.emplace_back( vector_to_matrix( flatFeatures ));
            labels.push_back( index );
        }

//        dlib::discriminant_pca dpca;

        _decisionFunction = trainer.train( featuresVector, labels );
    }


    std::vector<std::string> predict( const std::vector<std::string> &test ) const
    {
        std::vector<std::string> labels;
        for (auto &seq : test)
        {
            if( auto histograms = histogramsTrainer_( {seq} , _selectedKernels ); histograms  )
            {
                auto flatFeatures = MCOps::extractFlatFeatureVector( histograms.value() ,
                                                                     _selectedKernels );
                labels.push_back( _index2Label.at( _decisionFunction( vector_to_matrix( flatFeatures ))));
            }
            else labels.push_back(  "unclassified" );
        }
        return labels;
    }

private:
    void _featureSelection( const std::map<std::string, std::vector<std::string >> &training )
    {
        _selectedKernels = MCOps::withinJointAllUnionKernels( training, histogramsTrainer_ );
    }

    void _registerLabels( const std::map<std::string, std::vector<std::string >> &training )
    {
        _labels.clear();
        _label2Index.clear();
        _index2Label.clear();
        for (const auto &l : keys( training ))
            _labels.insert( l );

        int i = 0;
        for (auto &label : _labels)
        {
            _label2Index.emplace( label, i );
            _index2Label.emplace( i, label );
            ++i;
        }
    }

public:
    const HistogramsTrainer histogramsTrainer_;

private:
    std::set<std::string> _labels;
    std::map<std::string, int> _label2Index;
    std::unordered_map<int, std::string> _index2Label;
    Selection _selectedKernels;
//    std::vector<bool> _selectedFeaturesMask;

    DecisionFunction _decisionFunction;
};


#endif //MARKOVIAN_FEATURES_SVMMODEL_HPP
