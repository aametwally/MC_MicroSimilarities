//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMODEL_HPP


#include "common.hpp"


#include "dlib/svm_threaded.h"
#include "dlib_utilities.hpp"
#include "dlib/statistics/dpca.h"


class SVMModel
{
public:
    using SampleType = dlib::matrix<double, 0, 0>;
    using Label = int;
    static constexpr std::optional< double > Auto = std::nullopt;

private:
    using SVMBinaryKernel = dlib::radial_basis_kernel<SampleType>;
    using SVMRBFKernel = dlib::radial_basis_kernel<SampleType>;
    using SVMRBFTrainer = dlib::krr_trainer<SVMRBFKernel>;
    using SVMBinaryTrainer = dlib::krr_trainer<SVMBinaryKernel>;
    using SVMTrainer = dlib::one_vs_one_trainer<dlib::any_trainer<SampleType>, Label>;
    using DecisionFunction = SVMTrainer::trained_function_type;
public:
    explicit SVMModel( std::optional<double> lambda , std::optional<double> gamma = Auto );

    void fit( const std::vector<std::string_view > &labels , std::vector< std::vector< double >> &&featuresVector );
    std::string_view predict( const std::vector< double > &features) const;

private:

    static SampleType _svmFeatures( const std::vector< double > &features );
    static std::vector< SampleType > _svmFeatures( std::vector< std::vector< double >> &&features );
    std::vector< Label > _registerLabels(  const std::vector<std::string_view > &labels );

private:
    const std::optional<double> _gamma;
    const std::optional<double> _lambda;
    std::set<std::string_view> _labels;
    std::map<std::string_view, int> _label2Index;
    std::unordered_map<int, std::string_view> _index2Label;

    DecisionFunction _decisionFunction;

};


#endif //MARKOVIAN_FEATURES_SVMMODEL_HPP
