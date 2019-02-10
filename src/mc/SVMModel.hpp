//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_SVMMODEL_HPP
#define MARKOVIAN_FEATURES_SVMMODEL_HPP

// STL
#include <variant>

// dlib
#include "dlib/svm_threaded.h"
#include "dlib_utilities.hpp"
#include "dlib/statistics/dpca.h"

// local
#include "common.hpp"

#include "SimilarityMetrics.hpp"


struct SVMConfiguration
{
    enum class GammaMultiLabelSettingEnum
    {
        SingleGamma_ONE_VS_ALL, // 1 parameter
        SingleGamma_ONE_VS_ONE, // 1 parameter
        GammaVector_ONE_VS_ALL, // N parameters
        GammaVector_ONE_VS_ONE // N(N-1) parameters
    };

    struct Tuning
    {
        explicit Tuning(
                size_t trials,
                std::pair<double, double> bounds
        )
                : maxTrials( trials ),
                  gammaBounds( std::move( bounds ))
        {}

        std::pair<double, double> gammaBounds;
        size_t maxTrials;
    };

    static Tuning defaultTuning()
    {
        return Tuning( 90, {1e-5, 100} );
    }

    explicit SVMConfiguration(
            double g,
            std::optional<Tuning> tuning
    )
            : gamma( g ), tuning( std::move( tuning ))
    {}

    SVMConfiguration() = default;

    static SVMConfiguration defaultHyperParametersConfiguration()
    {
        return SVMConfiguration( 0.01, defaultTuning());
    }

    GammaMultiLabelSettingEnum gammaSetting;
    std::optional<std::vector<double>> gamma;
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

    using SVMOneVsOneTrainer =  dlib::one_vs_one_trainer<dlib::any_trainer<SampleType>, Label>;
    using SVMOneVsAllTrainer =  dlib::one_vs_all_trainer<dlib::any_trainer<SampleType>, Label>;

    using DecisionFunctionVariant = std::variant<
            SVMOneVsOneTrainer::trained_function_type,
            SVMOneVsAllTrainer::trained_function_type>;

    struct DecisionFunction
    {
        DecisionFunction( SVMOneVsOneTrainer::trained_function_type trainedFn )
                : _fn( std::move( trainedFn ))
        {}

        DecisionFunction( SVMOneVsAllTrainer::trained_function_type trainedFn )
                : _fn( std::move( trainedFn ))
        {}

        DecisionFunction() = default;

        DecisionFunction &operator=( SVMOneVsOneTrainer::trained_function_type trainedFn )
        {
            _fn = std::move( trainedFn );
        }

        int operator()( const SampleType &sample ) const
        {
            return std::visit( [&]( auto &&df ) -> int {
                return df( sample );
            }, _fn );
        }

    private:
        DecisionFunctionVariant _fn;
    };

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

    static auto _crossValidationScoreSingleGamma_ONE_VS_ALL(
            const std::vector<SVMModel::SampleType> &samples,
            const std::vector<SVMModel::Label> &labels
    );

    static auto _crossValidationScoreMultipleGammas_ONE_VS_ALL(
            const std::vector<SVMModel::SampleType> &samples,
            const std::vector<SVMModel::Label> &labels,
            const std::map<std::string_view, int> &label2Index
    );

    static auto _crossValidationScoreMultipleGammas_ONE_VS_ONE(
            const std::vector<SVMModel::SampleType> &samples,
            const std::vector<SVMModel::Label> &labels,
            const std::vector<std::pair<int, int>> &one2oneComb
    );

    static SampleType _svmFeatures( std::vector<double> &&features );

    static std::vector<SampleType> _svmSamples( std::vector<std::vector<double >> &&features );

    std::vector<Label> _registerLabels( std::vector<std::string_view> &&labels );

    static std::vector<std::pair<int, int>> _one2oneCombination( int nLabels );

private:
    const SVMConfiguration _configuration;
    std::set<std::string_view> _labels;
    std::map<std::string_view, int> _label2Index;
    std::unordered_map<int, std::string_view> _index2Label;
    DecisionFunction _decisionFunction;
};


#endif //MARKOVIAN_FEATURES_SVMMODEL_HPP
