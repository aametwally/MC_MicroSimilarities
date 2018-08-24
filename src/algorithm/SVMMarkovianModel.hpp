//
// Created by asem on 13/08/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMODEL_HPP

#include "common.hpp"
#include "MarkovianKernels.hpp"
#include "dlib/svm_threaded.h"
#include "dlib_utilities.hpp"
#include "dlib/statistics/dpca.h"

template<typename Grouping>
class SVMMarkovianModel
{
public:
    using MarkovianProfile = MarkovianKernels<Grouping>;
    using Kernel = typename MarkovianProfile::Kernel;
    using MarkovianProfiles = std::map<std::string, MarkovianProfile>;
    using KernelID = typename MarkovianProfile::KernelID;
    using Order = typename MarkovianProfile::Order;
    using KernelsSet = std::unordered_map<Order, std::set<KernelID >>;

    using HeteroKernels =  typename MarkovianProfile::HeteroKernels;
    using HeteroKernelsFeatures =  typename MarkovianProfile::HeteroKernelsFeatures;

    using DoubleSeries = typename MarkovianProfile::ProbabilitisByOrder;
    using KernelsSeries = typename MarkovianProfile::KernelSeriesByOrder;

    static constexpr Order MinOrder = MarkovianProfile::MinOrder;
    static constexpr size_t StatesN = MarkovianProfile::StatesN;
    static constexpr double eps = std::numeric_limits<double>::epsilon();

    using SampleType = dlib::matrix<double , 0 , 0 >;

    using SVMBinaryKernel = dlib::radial_basis_kernel<SampleType>;
    using Label = int;

    using SVMRBFKernel = dlib::radial_basis_kernel<SampleType>;
    using SVMRBFTrainer = dlib::krr_trainer<SVMRBFKernel>;
    using SVMBinaryTrainer = dlib::krr_trainer<SVMBinaryKernel >;
    using SVMTrainer = dlib::one_vs_one_trainer<dlib::any_trainer<SampleType> , Label >;
    using DecisionFunction = SVMTrainer::trained_function_type;

public:
    explicit SVMMarkovianModel( Order maxOrder )
            : maxOrder_( maxOrder )
    {}


    void fit( const std::map<std::string, std::vector<std::string >> &training )
    {

        SVMTrainer trainer;
        SVMBinaryTrainer btrainer;
//        histogramTrainer.set_lambda(0.00001);
        btrainer.set_kernel( SVMBinaryKernel( 0.01 ));
//        histogramTrainer.set_max_num_sv(10);

        trainer.set_trainer(btrainer );
        trainer.set_num_threads( std::thread::hardware_concurrency());

        _registerLabels( training );
        _featureSelection( training );

        std::vector<int> labels;
        std::vector<SampleType> featuresVector;
        for (auto &[trainLabel, trainSeqs] : training)
        {
            int index = _label2Index.at( trainLabel );
            MarkovianProfile kernels( maxOrder_ );
            kernels.train( trainSeqs );
            auto flatFeatures = kernels.extractFlatFeatureVector( _selectedKernels );
            featuresVector.emplace_back( vector_to_matrix( flatFeatures ));
            labels.push_back( index );
        }

//        dlib::discriminant_pca dpca;

        _decisionFunction = trainer.train( featuresVector, labels );
    }


    std::vector<std::string> predict( const std::vector<std::string> &test ) const
    {
        std::vector<std::string> labels;
        std::vector<SampleType> featuresVector;
        for (auto &seq : test)
        {

            MarkovianProfile kernels( maxOrder_ );
            kernels.train( {seq} );

            auto flatFeatures = kernels.extractFlatFeatureVector( _selectedKernels );

            labels.push_back( _index2Label.at( _decisionFunction( vector_to_matrix( flatFeatures ))));

        }

        return labels;
    }

private:
    void _featureSelection( const std::map<std::string, std::vector<std::string >> &training )
    {
        std::tie( std::ignore ,  _selectedKernels) = MarkovianProfile::coveredFeatures( training, maxOrder_ );
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
    const Order maxOrder_;

private:
    std::set<std::string> _labels;
    std::map<std::string, int> _label2Index;
    std::unordered_map<int, std::string> _index2Label;
    KernelsSet _selectedKernels;
//    std::vector<bool> _selectedFeaturesMask;

    DecisionFunction _decisionFunction;
};


#endif //MARKOVIAN_FEATURES_SVMMODEL_HPP
