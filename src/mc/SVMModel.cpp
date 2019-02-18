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

void SVMModel::fit(
        std::vector<std::string_view> &&labels,
        std::vector<std::vector<double >> &&samples
)
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
        const std::map<std::string_view, int> &label2Index
)
{
    if ( configuration.tuning )
    {
        return _fitTuningHyperParameters(
                std::move( labels ), std::move( samples ), configuration, label2Index );
    } else
    {
        assert( configuration.gamma );
        return _fitFixedHyperParameters(
                std::move( labels ), std::move( samples ), configuration, label2Index );
    }
}

ScoredLabels SVMModel::predict( std::vector<double> &&features ) const
{
    auto bestLabel = _index2Label.at( _decisionFunction( _svmFeatures( std::move( features ))));
    std::map<std::string_view, size_t> votes;
    for ( auto &[label, idx] : _label2Index )
        votes[label] += 0;
    votes[bestLabel] += 1;

    ScoredLabels vPQ( _label2Index.size());
    for ( auto&[label, n] : votes )
        vPQ.emplace( label, n );
    return vPQ;
}

SVMModel::SampleType SVMModel::_svmFeatures( std::vector<double> &&features )
{
    using namespace dlib_utilities;
    return vector_to_column_matrix_like( std::move( features ));
}

std::vector<SVMModel::SampleType> SVMModel::_svmSamples( std::vector<std::vector<double >> &&samples )
{
    std::vector<SampleType> svmSamples;
    for ( auto &&f : samples )
        svmSamples.emplace_back( _svmFeatures( std::move( f )));
    return svmSamples;
}

std::vector<std::pair<int, int>> SVMModel::_one2oneCombination( int nLabels )
{
    std::vector<std::pair<int, int>> pairs;
    for ( auto i = 0; i < nLabels; ++i )
    {
        for ( auto j = i + 1; j < nLabels; ++j )
        {
            pairs.emplace_back( i, j );
        }
    }
    return pairs;
}

std::vector<SVMModel::Label> SVMModel::_registerLabels( std::vector<std::string_view> &&labels )
{
    std::vector<Label> svmLabels;

    _labels.clear();
    _label2Index.clear();
    _index2Label.clear();
    for ( auto label : labels )
        _labels.insert( label );

    int i = 0;
    for ( auto &label : _labels )
    {
        _label2Index.emplace( label, i );
        _index2Label.emplace( i, label );
        ++i;
    }

    for ( auto label : labels )
        svmLabels.push_back( _label2Index.at( label ));
    return svmLabels;
}

SVMModel::DecisionFunction SVMModel::_fitFixedHyperParameters(
        std::vector<Label> &&labels,
        std::vector<SampleType> &&samples,
        const SVMConfiguration &configuration,
        const std::map<std::string_view, int> &label2Index
)
{
    using GammaSetting = SVMConfiguration::GammaMultiLabelSettingEnum;
    assert( configuration.gamma );
    if ( !configuration.gamma )
        throw std::runtime_error( "Missing gamma value." );
    auto &&gamma = configuration.gamma.value();

    switch ( configuration.gammaSetting )
    {
        case GammaSetting::SingleGamma_ONE_VS_ALL :
        {
            assert( gamma.size() == 1 );
            if ( gamma.size() != 1 )
                throw std::runtime_error( "Gamma vector size must equal 1." );

            SVMOneVsAllTrainer trainer;
            trainer.set_num_threads( std::thread::hardware_concurrency());
            SVMBinaryTrainer btrainer;
            btrainer.use_classification_loss_for_loo_cv();
            btrainer.set_kernel( SVMRBFKernel( gamma.front()));
            trainer.set_trainer( btrainer );
            return trainer.train( samples, labels );
        }
            break;
        case GammaSetting::GammaVector_ONE_VS_ALL:
        {
            assert( gamma.size() == label2Index.size());
            if ( gamma.size() != label2Index.size())
                throw std::runtime_error( "Gammas count should equal to labels count." );

            SVMOneVsAllTrainer trainer;
            trainer.set_num_threads( std::thread::hardware_concurrency());

            for ( auto i = 0; i < label2Index.size(); ++i )
            {
                SVMBinaryTrainer btrainer;
                btrainer.use_classification_loss_for_loo_cv();
                btrainer.set_kernel( SVMRBFKernel( gamma.at( i )));
                btrainer.be_verbose();
                trainer.set_trainer( btrainer, i );
                trainer.be_verbose();
            }
            return trainer.train( samples, labels );
        }
            break;
        case GammaSetting::GammaVector_ONE_VS_ONE:
        {
            auto nLabels = label2Index.size();
            auto one2oneComb = _one2oneCombination( static_cast<int>(nLabels));
            assert( gamma.size() == nLabels * ( nLabels - 1 ) / 2 &&
                    gamma.size() == one2oneComb.size());
            if ( gamma.size() != nLabels * ( nLabels - 1 ) / 2
                 || gamma.size() != one2oneComb.size())
                throw std::runtime_error( "Gammas count should equal to n(n-1)/2; n: labeles count." );

            SVMOneVsOneTrainer trainer;
            trainer.set_num_threads( std::thread::hardware_concurrency());

            for ( auto i = 0; i < one2oneComb.size(); ++i )
            {
                SVMBinaryTrainer btrainer;
                btrainer.use_classification_loss_for_loo_cv();
                btrainer.set_kernel( SVMRBFKernel( gamma.at( i )));
                trainer.set_trainer( btrainer, one2oneComb.at( i ).first, one2oneComb.at( i ).second );
            }
            return trainer.train( samples, labels );
        }
            break;
        default:throw std::runtime_error( "Unhandled gamma setting" );
    }

    throw std::runtime_error( "Function should not reach here." );
}

auto SVMModel::_crossValidationScoreSingleGamma_ONE_VS_ALL(
        const std::vector<SVMModel::SampleType> &samples,
        const std::vector<SVMModel::Label> &labels
)
{
    return [&]( const double gamma )
    {
      SVMBinaryTrainer btrainer;
      btrainer.use_classification_loss_for_loo_cv();

      SVMOneVsAllTrainer trainer;
      trainer.set_num_threads( 1 );
      btrainer.set_kernel( SVMRBFKernel( gamma ));
      trainer.set_trainer( btrainer );

      // Finally, perform 10-fold cross validation and then print and return the results.
      auto result = dlib::cross_validate_multiclass_trainer( trainer, samples, labels, 10 );
      auto rawCM = std::vector<std::vector<size_t>>( result.nr(), std::vector<size_t>( result.nc()));
      for ( auto r = 0; r < result.nr(); ++r )
          for ( auto c = 0; c < result.nc(); ++c )
              rawCM[r][c] = static_cast<size_t>(result( r, c ));

      auto cm = ConfusionMatrix<std::string, size_t>::fromRawConfusionMatrix( rawCM );
      auto objective = cm.microFScore();

//        fmt::print( "gamma:{:.11f}\n"
//                    "CV F1-Score:{}\n", gamma, objective );
      return objective;
    };
}

auto SVMModel::_crossValidationScoreMultipleGammas_ONE_VS_ALL(
        const std::vector<SVMModel::SampleType> &samples,
        const std::vector<SVMModel::Label> &labels,
        const std::map<std::string_view, int> &label2Index
)
{
    return [&]( dlib::matrix<double, 0, 1> gammas )
    {
      assert( gammas.size() == label2Index.size());
      SVMOneVsAllTrainer trainer;
      trainer.set_num_threads( 1 );
      for ( auto &&[label, index] : label2Index )
      {
          SVMBinaryTrainer btrainer;
          btrainer.use_classification_loss_for_loo_cv();
          btrainer.set_kernel( SVMRBFKernel( gammas( index )));
          trainer.set_trainer( btrainer, index );
      }

      // Finally, perform 10-fold cross validation and then print and return the results.
      auto result = dlib::cross_validate_multiclass_trainer( trainer, samples, labels, 10 );
      auto rawCM = std::vector<std::vector<size_t>>( result.nr(), std::vector<size_t>( result.nc()));
      for ( auto r = 0; r < result.nr(); ++r )
          for ( auto c = 0; c < result.nc(); ++c )
              rawCM[r][c] = static_cast<size_t>(result( r, c ));

      auto cm = ConfusionMatrix<std::string, size_t>::fromRawConfusionMatrix( rawCM );
      auto objective = cm.microFScore();
//        fmt::print( "ovaGammas:{:.11f}\n"
//                    "CV F1-Score:{}\n", fmt::join( ovaGammas, ", " ), objective );
      return objective;
    };
}

auto SVMModel::_crossValidationScoreMultipleGammas_ONE_VS_ONE(
        const std::vector<SVMModel::SampleType> &samples,
        const std::vector<SVMModel::Label> &labels,
        const std::vector<std::pair<int, int>> &one2oneComb
)
{
    return [&]( dlib::matrix<double, 0, 1> gammas )
    {
      assert( gammas.size() == one2oneComb.size());
      SVMOneVsOneTrainer trainer;
      trainer.set_num_threads( 1 );
      for ( auto i = 0; i < one2oneComb.size(); ++i )
      {
          SVMBinaryTrainer btrainer;
          btrainer.use_classification_loss_for_loo_cv();
          btrainer.set_kernel( SVMRBFKernel( gammas( i )));
          trainer.set_trainer( btrainer, one2oneComb.at( i ).first, one2oneComb.at( i ).second );
      }

      // Finally, perform 10-fold cross validation and then print and return the results.
      auto result = dlib::cross_validate_multiclass_trainer( trainer, samples, labels, 10 );
      auto rawCM = std::vector<std::vector<size_t>>( result.nr(), std::vector<size_t>( result.nc()));
      for ( auto r = 0; r < result.nr(); ++r )
          for ( auto c = 0; c < result.nc(); ++c )
              rawCM[r][c] = static_cast<size_t>(result( r, c ));

      auto cm = ConfusionMatrix<std::string, size_t>::fromRawConfusionMatrix( rawCM );
      auto objective = cm.microFScore();
//        fmt::print( "ovoGammas:{:.11f}\n"
//                    "CV F1-Score:{}\n" , fmt::join( gammas , ", " ) , objective );
      return objective;
    };
}

SVMModel::DecisionFunction SVMModel::_fitTuningHyperParameters(
        std::vector<Label> &&labels,
        std::vector<SampleType> &&samples,
        const SVMConfiguration &configuration,
        const std::map<std::string_view, int> &label2Index
)
{
    using GammaSetting = SVMConfiguration::GammaMultiLabelSettingEnum;

    assert( configuration.tuning );
    auto newConfig = configuration;
    newConfig.tuning.reset();
    newConfig.gamma.reset();

    auto tp = dlib::thread_pool(
            static_cast<size_t>(std::max( 1, static_cast<int>(std::thread::hardware_concurrency()))));
    auto[min, max] = configuration.tuning->gammaBounds;
    auto maxCalls = configuration.tuning->maxTrials;

    switch ( configuration.gammaSetting )
    {
        case GammaSetting::SingleGamma_ONE_VS_ALL :
        {
            using DLIBCVector = dlib::matrix<double, 0, 1>;

            DLIBCVector minGammas = dlib::uniform_matrix<double>( 1, 1, min );
            DLIBCVector maxGammas = dlib::uniform_matrix<double>( 1, 1, max );

            auto result = dlib::find_max_global(
                    tp, _crossValidationScoreSingleGamma_ONE_VS_ALL( samples, labels ),
                    minGammas, maxGammas, dlib::max_function_calls( maxCalls ));

            newConfig.gamma.emplace( {result.x( 0 )} );
        }
            break;
        case GammaSetting::GammaVector_ONE_VS_ALL:
        {
            using DLIBCVector = dlib::matrix<double, 0, 1>;

            DLIBCVector minGammas = dlib::uniform_matrix<double>( label2Index.size(), 1, min );
            DLIBCVector maxGammas = dlib::uniform_matrix<double>( label2Index.size(), 1, max );

            auto result = dlib::find_max_global(
                    tp, _crossValidationScoreMultipleGammas_ONE_VS_ALL(
                            samples, labels, label2Index ), minGammas, maxGammas,
                    dlib::max_function_calls( static_cast<size_t>( maxCalls )));

            auto &&gammas = result.x;
            std::vector<double> newGammas;
            for ( double gamma : gammas )
                newGammas.push_back( gamma );

            newConfig.gamma.emplace( std::move( newGammas ));
        }
            break;
        case GammaSetting::GammaVector_ONE_VS_ONE:
        {
            int nLabels = static_cast<int>( label2Index.size());
            auto one2oneComb = _one2oneCombination( nLabels );
            using DLIBCVector = dlib::matrix<double, 0, 1>;

            DLIBCVector minGammas = dlib::uniform_matrix<double>( one2oneComb.size(), 1, min );
            DLIBCVector maxGammas = dlib::uniform_matrix<double>( one2oneComb.size(), 1, max );

            auto result = dlib::find_max_global(
                    tp, _crossValidationScoreMultipleGammas_ONE_VS_ONE(
                            samples, labels, one2oneComb ), minGammas, maxGammas,
                    dlib::max_function_calls( static_cast<size_t>( maxCalls )));

            auto &&gammas = result.x;
            std::vector<double> newGammas;
            for ( double gamma : gammas )
                newGammas.push_back( gamma );

            newConfig.gamma.emplace( std::move( newGammas ));
        }
            break;
        default:throw std::runtime_error( "Unhandled gamma setting" );
    }

    return _fit( std::move( labels ), std::move( samples ), newConfig, label2Index );
}
