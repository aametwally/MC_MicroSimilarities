//
// Created by asem on 13/09/18.
//
#include "SVMModel.hpp"

SVMModel::SVMModel( std::optional<double> lambda , std::optional<double> gamma )
        : _lambda( lambda ), _gamma( gamma )
{

}


void SVMModel::fit( const std::vector<std::string_view> &labels, std::vector<std::vector<double >> &&featuresVector )
{
    auto svmLabels = _registerLabels( labels );
    auto nFeatures = featuresVector.front().size();
    auto svmFeatures = _svmFeatures( std::move( featuresVector ));

    SVMTrainer trainer;
    SVMBinaryTrainer btrainer;
    btrainer.set_lambda( (_lambda)? _lambda.value() : 1.0 );
    btrainer.set_kernel( SVMBinaryKernel( (_gamma)? _gamma.value() : 1.0 / nFeatures ));
//        histogramTrainer.set_max_num_sv(10);

    trainer.set_trainer( btrainer );
    trainer.set_num_threads( std::thread::hardware_concurrency());

    _decisionFunction = trainer.train( svmFeatures, svmLabels );
}

std::string_view SVMModel::predict( const std::vector<double> &features ) const
{
    return _index2Label.at( _decisionFunction( _svmFeatures( features )));
}


SVMModel::SampleType SVMModel::_svmFeatures( const std::vector<double> &features )
{
    return vector_to_matrix( features );
}

std::vector<SVMModel::SampleType> SVMModel::_svmFeatures( std::vector<std::vector<double >> &&features )
{
    std::vector<SampleType> svmFeatures;
    for (auto &f : features)
        svmFeatures.emplace_back( _svmFeatures( f ));
    return svmFeatures;
}

std::vector<SVMModel::Label> SVMModel::_registerLabels( const std::vector<std::string_view> &labels )
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

