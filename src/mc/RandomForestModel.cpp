//
// Created by asem on 10/02/19.
//

#include "RandomForestModel.hpp"
#include "dlib_utilities.hpp"

RandomForestModel::RandomForestModel( RandomForestConfiguration config )
        : _config( config )
{

}

void RandomForestModel::fit(
        std::vector<std::string_view> &&labels,
        std::vector<std::vector<double >> &&featuresVector
)
{
    _decisionFunction =
            _fit( _registerLabels( std::move( labels )),
                  _dlibSamples( std::move( featuresVector )),
                  _config,
                  _label2Index );

}

ScoredLabels RandomForestModel::predict( std::vector<double> &&features ) const
{
    auto bestLabel = _index2Label.at( _decisionFunction( _dlibFeatures( std::move( features ))));
    std::map<std::string_view, size_t> votes;
    for ( auto &[label, idx] : _label2Index )
        votes[label] += 0;
    votes[bestLabel] += 1;

    ScoredLabels vPQ( _label2Index.size());
    for ( auto&[label, n] : votes )
        vPQ.emplace( label, n );
    return vPQ;
}

RandomForestModel::SampleType RandomForestModel::_dlibFeatures( std::vector<double> &&features )
{
    using namespace dlib_utilities;
    return vector_to_column_matrix_like( std::move( features ));
}

std::vector<RandomForestModel::SampleType>
RandomForestModel::_dlibSamples( std::vector<std::vector<double >> &&features )
{
    std::vector<SampleType> dlibSamples;
    for ( auto &&f : features )
        dlibSamples.emplace_back( _dlibFeatures( std::move( f )));
    return dlibSamples;
}

std::vector<RandomForestModel::Label> RandomForestModel::_registerLabels( std::vector<std::string_view> &&labels )
{
    std::vector<Label> dlibLabels;

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
        dlibLabels.push_back( _label2Index.at( label ));
    return dlibLabels;
}

RandomForestModel::DecisionFunction RandomForestModel::_fit(
        std::vector<RandomForestModel::Label> &&labels, std::vector<RandomForestModel::SampleType> &&samples,
        const RandomForestConfiguration &configuration,
        const std::map<std::string_view, RandomForestModel::Label> &label2Index
)
{
    RFTrainer btrainer;
    btrainer.set_num_trees( configuration.nTrees );
    btrainer.set_seed( "random forest" );
    btrainer.be_verbose();

    SVMOneVsAllTrainer trainer;
    trainer.set_trainer( btrainer );
    trainer.set_num_threads( 1 );
    trainer.be_verbose();
    return trainer.train( samples, labels );
}
