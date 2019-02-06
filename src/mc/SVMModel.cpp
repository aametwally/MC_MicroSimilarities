//
// Created by asem on 13/09/18.
//
#include "SVMModel.hpp"
#include <dlib/global_optimization.h>
#include "ConfusionMatrix.hpp"

SVMModel::SVMModel( SVMConfiguration config )
        : _configuration( std::move( config ))
{

}

void SVMModel::fit( std::vector<std::string_view> &&labels, std::vector<std::vector<double >> &&samples )
{
    _decisionFunction =
            _fit( _registerLabels( std::move( labels )),
                  _svmSamples( std::move( samples )),
                  _configuration,
                  _label2Index );
}

SVMModel::DecisionFunction SVMModel::_fit(
        std::vector<Label> &&labels,
        std::vector<SampleType> &&samples,
        const SVMConfiguration &configuration,
        const std::map<std::string_view, int> &label2Index )
{
    if ( configuration.tuning )
    {
        return _fitTuningHyperParameters(
                std::move( labels ), std::move( samples ), configuration, label2Index );
    } else
    {
        assert( configuration.gammas || configuration.gamma );
        return _fitFixedHyperParameters(
                std::move( labels ), std::move( samples ), configuration, label2Index );
    }
}

ScoredLabels SVMModel::predict( std::vector<double> &&features ) const
{
    auto bestLabel = _index2Label.at( _decisionFunction( _svmFeatures( std::move( features ))));
    std::map<std::string_view, size_t> votes;
    for (auto &[label, idx] : _label2Index)
        votes[label] += 0;
    votes[bestLabel] += 1;

    ScoredLabels vPQ( _label2Index.size());
    for (auto&[label, n] : votes)
        vPQ.emplace( label, n );
    return vPQ;
}

SVMModel::SampleType SVMModel::_svmFeatures( std::vector<double> &&features )
{
    using namespace dlib_utilities;
    return vectorToColumnMatrixLike( std::move( features ));
}

std::vector<SVMModel::SampleType> SVMModel::_svmSamples( std::vector<std::vector<double >> &&samples )
{
    std::vector<SampleType> svmSamples;
    for (auto &&f : samples)
        svmSamples.emplace_back( _svmFeatures( std::move( f )));
    return svmSamples;
}

std::vector<SVMModel::Label> SVMModel::_registerLabels( std::vector<std::string_view> &&labels )
{
    std::vector<Label> svmLabels;

    _labels.clear();
    _label2Index.clear();
    _index2Label.clear();
    for (auto label : labels)
        _labels.insert( label );

    int i = 0;
    for (auto &label : _labels)
    {
        _label2Index.emplace( label, i );
        _index2Label.emplace( i, label );
        ++i;
    }

    for (auto label : labels)
        svmLabels.push_back( _label2Index.at( label ));
    return svmLabels;
}

SVMModel::DecisionFunction SVMModel::_fitFixedHyperParameters(
        std::vector<Label> &&labels,
        std::vector<SampleType> &&samples,
        const SVMConfiguration &configuration,
        const std::map<std::string_view, int> &label2Index )
{
    SVMTrainer trainer;
    trainer.set_num_threads( std::thread::hardware_concurrency());

    if ( configuration.gamma )
    {
        SVMBinaryTrainer btrainer;
        btrainer.use_classification_loss_for_loo_cv();
        btrainer.set_kernel( SVMRBFKernel( configuration.gamma.value()));
        trainer.set_trainer( btrainer );
    } else if ( configuration.gammas )
    {
        assert( configuration.gammas->size() == label2Index.size());
        if ( configuration.gammas->size() != label2Index.size())
            throw std::runtime_error( "Gammas count should equal to labels count." );

        for (auto &&[label, gamma] : configuration.gammas.value())
        {
            SVMBinaryTrainer btrainer;
            btrainer.use_classification_loss_for_loo_cv();
            btrainer.set_kernel( SVMRBFKernel( gamma ));
            trainer.set_trainer( btrainer, label2Index.at( label ));
        }
    } else throw std::runtime_error( "Gammas are not defined!" );
    return trainer.train( samples, labels );
}

auto SVMModel::_crossValidationScoreSingleGamma(
        const std::vector<SVMModel::SampleType> &samples,
        const std::vector<SVMModel::Label> &labels )
{
    return [&]( const double gamma ) {
        SVMBinaryTrainer btrainer;
        btrainer.use_classification_loss_for_loo_cv();

        SVMTrainer trainer;

        btrainer.set_kernel( SVMRBFKernel( gamma ));
        trainer.set_trainer( btrainer );

        // Finally, perform 10-fold cross validation and then print and return the results.
        auto result = dlib::cross_validate_multiclass_trainer( trainer, samples, labels, 10 );
        auto rawCM = std::vector<std::vector<size_t>>( result.nr(), std::vector<size_t>( result.nc()));
        for (auto r = 0; r < result.nr(); ++r)
            for (auto c = 0; c < result.nc(); ++c)
                rawCM[r][c] = static_cast<size_t>(result( r, c ));

        auto cm = ConfusionMatrix<std::string, size_t>::fromRawConfusionMatrix( rawCM );

        fmt::print( "gamma:{:.11f}\n"
                    "CV accuracy:{}\n", gamma, result );

        return cm.overallAccuracy();
    };
}

auto SVMModel::_crossValidationScoreMultipleGammas(
        const std::vector<SVMModel::SampleType> &samples,
        const std::vector<SVMModel::Label> &labels,
        const std::map<std::string_view, int> &label2Index )
{
    return [&]( dlib::matrix<double, 0, 1> gammas ) {
        assert( gammas.size() == label2Index.size());
        SVMTrainer trainer;
        for (auto &&[label, index] : label2Index)
        {
            SVMBinaryTrainer btrainer;
            btrainer.use_classification_loss_for_loo_cv();
            btrainer.set_kernel( SVMRBFKernel( gammas( index )));
            trainer.set_trainer( btrainer, index );
        }

        // Finally, perform 10-fold cross validation and then print and return the results.
        auto result = dlib::cross_validate_multiclass_trainer( trainer, samples, labels, 10 );
        auto rawCM = std::vector<std::vector<size_t>>( result.nr(), std::vector<size_t>( result.nc()));
        for (auto r = 0; r < result.nr(); ++r)
            for (auto c = 0; c < result.nc(); ++c)
                rawCM[r][c] = static_cast<size_t>(result( r, c ));

        auto cm = ConfusionMatrix<std::string, size_t>::fromRawConfusionMatrix( rawCM );
        fmt::print( "gamma:{:.11f}\n"
                    "CV accuracy:{}\n", fmt::join( gammas, ", " ), result );

        return cm.overallAccuracy();
    };
}


SVMModel::DecisionFunction SVMModel::_fitTuningHyperParameters(
        std::vector<Label> &&labels,
        std::vector<SampleType> &&samples,
        const SVMConfiguration &configuration,
        const std::map<std::string_view, int> &label2Index )
{
    assert( configuration.tuning );
    auto newConfig = configuration;
    newConfig.tuning.reset();
    newConfig.gamma.reset();
    newConfig.gammas.reset();

    auto tp = dlib::thread_pool( std::thread::hardware_concurrency());
    auto[min, max] = configuration.tuning->gammaBounds;
    auto maxCalls = configuration.tuning->maxTrials;

    if ( configuration.tuning->tuneGammaPerClass )
    {
        dlib::matrix<double, 0, 1> minMat = dlib_utilities::vectorToColumnMatrixLike(
                std::vector<double>( label2Index.size(), min ));
        dlib::matrix<double, 0, 1> maxMat = dlib_utilities::vectorToColumnMatrixLike(
                std::vector<double>( label2Index.size(), max ));

        auto result = dlib::find_max_global(
                tp, _crossValidationScoreMultipleGammas( samples, labels, label2Index ), minMat, maxMat,
                dlib::max_function_calls( static_cast<size_t>(maxCalls * label2Index.size())));

        std::map<std::string_view, double> newGammas;
        for (auto &&[label, index] : label2Index)
        {
            newGammas.emplace( label, result.x( index ));
        }
        newConfig.gammas.emplace( std::move( newGammas ));
    } else
    {
        dlib::matrix<double, 0, 1> minMat = dlib_utilities::vectorToColumnMatrixLike( {min} );
        dlib::matrix<double, 0, 1> maxMat = dlib_utilities::vectorToColumnMatrixLike( {max} );

        auto result = dlib::find_max_global(
                tp, _crossValidationScoreSingleGamma( samples, labels ),
                minMat, maxMat, dlib::max_function_calls( maxCalls ));

        newConfig.gamma.emplace( result.x( 0 ));
    }

    return _fit( std::move( labels ), std::move( samples ), newConfig, label2Index );
}
