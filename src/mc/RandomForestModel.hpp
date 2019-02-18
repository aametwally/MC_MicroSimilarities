//
// Created by asem on 10/02/19.
//

#ifndef MARKOVIAN_FEATURES_RANDOMFORESTMODEL_HPP
#define MARKOVIAN_FEATURES_RANDOMFORESTMODEL_HPP

// STL
#include <variant>

// dlib
#include <dlib/random_forest.h>
#include <dlib/svm.h>
#include <dlib/statistics.h>
#include <dlib/svm_threaded.h>

// local
#include "common.hpp"
#include "SimilarityMetrics.hpp"

struct RandomForestConfiguration
{
    size_t nTrees = 1000;
};

class RandomForestModel
{
    using SampleType = dlib::matrix<double, 0, 1>;
    using Label = double;
    using FeatureExtractor = dlib::dense_feature_extractor;
    using RFTrainer = dlib::random_forest_regression_trainer<FeatureExtractor>;
    using SVMOneVsAllTrainer =  dlib::one_vs_all_trainer<dlib::any_trainer<SampleType>, Label>;
    using DecisionFunction = SVMOneVsAllTrainer::trained_function_type;

public:
    explicit RandomForestModel( RandomForestConfiguration config );

    virtual ~RandomForestModel() = default;

    void fit(
            std::vector<std::string_view> &&labels,
            std::vector<std::vector<double >> &&featuresVector
    );

    ScoredLabels predict( std::vector<double> &&features ) const;


private:
    DecisionFunction _fit(
            std::vector<Label> &&labels,
            std::vector<SampleType> &&samples,
            const RandomForestConfiguration &configuration,
            const std::map<std::string_view, Label> &label2Index
    );

    static SampleType _dlibFeatures( std::vector<double> &&features );

    static std::vector<SampleType> _dlibSamples( std::vector<std::vector<double >> &&features );

    std::vector<Label> _registerLabels( std::vector<std::string_view> &&labels );

private:
    RandomForestConfiguration _config;
    std::set<std::string_view> _labels;
    std::map<std::string_view, Label> _label2Index;
    std::unordered_map<Label, std::string_view> _index2Label;
    DecisionFunction _decisionFunction;
};


#endif //MARKOVIAN_FEATURES_RANDOMFORESTMODEL_HPP
