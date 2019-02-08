//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMODEL_HPP


#include "common.hpp"


#include "dlib/svm_threaded.h"
#include "dlib_utilities.hpp"
#include "dlib/statistics/dpca.h"

#include "SimilarityMetrics.hpp"


struct SVMConfiguration
{
    struct Tuning
    {
        explicit Tuning(
                size_t trials,
                std::pair<double, double> bounds,
                bool tuneGammaPerClass
        )
                : maxTrials( trials ),
                  gammaBounds( std::move( bounds )),
                  tuneGammaPerClass( tuneGammaPerClass )
        {}

        std::pair<double, double> gammaBounds;
        bool tuneGammaPerClass;
        size_t maxTrials;
    };

    static Tuning defaultTuning()
    {
        return Tuning( 50, {1e-5, 100}, false );
    }

    explicit SVMConfiguration(
            double g,
            std::optional<Tuning> tuning
    )
            : gamma( g ), tuning( std::move( tuning ))
    {}

    explicit SVMConfiguration(
            std::map<std::string_view, double> gammas,
            std::optional<Tuning> tuning
    )
            : gammas( std::move( gammas )), tuning( std::move( tuning ))
    {}

    SVMConfiguration() = default;

    static SVMConfiguration defaultHyperParametersConfiguration()
    {
        return SVMConfiguration( 0.01, defaultTuning());
    }

    std::optional<double> gamma;
    std::optional<std::map<std::string_view, double>> gammas;
    std::optional<Tuning> tuning;
};

class SVMModel
{
public:
    using SampleType = dlib::matrix<double, 0, 1>;
    using Label = int;
    static constexpr std::optional<double> Auto = std::nullopt;

private:
    using SVMRBFKernel = dlib::radial_basis_kernel<SampleType>;
    //using SVMLinearKernel = dlib::linear_kernel<SampleType>;

    //using SVMMultiTrainer = dlib::svm_multiclass_linear_trainer<SVMLinearKernel>;
    using SVMBinaryTrainer = dlib::krr_trainer<SVMRBFKernel>;

    using SVMTrainer = dlib::one_vs_all_trainer<dlib::any_trainer<SampleType>, Label>;
    using DecisionFunction = SVMTrainer::trained_function_type;

public:
    explicit SVMModel( SVMConfiguration config );

    virtual ~SVMModel() = default;

    void fit(
            std::vector<std::string_view> &&labels,
            std::vector<std::vector<double >> &&featuresVector
    );

    ScoredLabels predict( std::vector<double> &&features ) const;

private:
    static DecisionFunction _fit(
            std::vector<Label> &&labels,
            std::vector<SampleType> &&featuresVector,
            const SVMConfiguration &configuration,
            const std::map<std::string_view, int> &label2Index
    );

    static DecisionFunction _fitFixedHyperParameters(
            std::vector<Label> &&labels,
            std::vector<SampleType> &&featuresVector,
            const SVMConfiguration &configuration,
            const std::map<std::string_view, int> &label2Index
    );

    static DecisionFunction _fitTuningHyperParameters(
            std::vector<Label> &&labels,
            std::vector<SampleType> &&featuresVector,
            const SVMConfiguration &configuration,
            const std::map<std::string_view, int> &label2Index
    );

    static auto _crossValidationScoreSingleGamma(
            const std::vector<SVMModel::SampleType> &samples,
            const std::vector<SVMModel::Label> &labels
    );

    static auto _crossValidationScoreMultipleGammas(
            const std::vector<SVMModel::SampleType> &samples,
            const std::vector<SVMModel::Label> &labels,
            const std::map<std::string_view, int> &label2Index
    );

    static SampleType _svmFeatures( std::vector<double> &&features );

    static std::vector<SampleType> _svmSamples( std::vector<std::vector<double >> &&features );

    std::vector<Label> _registerLabels( std::vector<std::string_view> &&labels );


private:
    const SVMConfiguration _configuration;
    std::set<std::string_view> _labels;
    std::map<std::string_view, int> _label2Index;
    std::unordered_map<int, std::string_view> _index2Label;
    DecisionFunction _decisionFunction;
};


#endif //MARKOVIAN_FEATURES_SVMMODEL_HPP
