//
// Created by asem on 02/01/19.
//

#ifndef MARKOVIAN_FEATURES_MCDISCRETIZEDSCALESCLASSIFIER_HPP
#define MARKOVIAN_FEATURES_MCDISCRETIZEDSCALESCLASSIFIER_HPP

#include "AAIndexClustering.hpp"

#include "AbstractClassifier.hpp"
#include "AbstractMC.hpp"
#include "ZYMC.hpp"
#include "MC.hpp"

namespace MC
{

const std::map<std::string , aaindex::AAIndex1> &indices = aaindex::extractAAIndices();

const std::vector<aaindex::AAIndex1> allIndices = []
{
    std::vector<aaindex::AAIndex1> selection;
    for ( auto &[label , index] : indices )
        if ( !index.hasMissingValues())
            selection.push_back( index );
    return selection;
}();

const std::vector<aaindex::AAIndex1> selectedIndices = []
{
    std::vector<aaindex::AAIndex1> selection;
    for ( auto &label : aaindex::ATCHLEY_FACTORS_MAX_CORRELATED_INDICES )
    {
        auto &index = indices.at( label );
        if ( !index.hasMissingValues())
            selection.emplace_back( index );
    }
    return selection;
}();

template < size_t States >
class MCDiscretizedScalesClassifier : public AbstractClassifier
{
    using MCModel = AbstractMC<States>;
    using ZMC = ZYMC<States>;
    using RMC = MC<States>;
    using MG = ModelGenerator<States>;

    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using ScoringFunction = std::function<double( std::string_view )>;

    static constexpr std::array<char , States> alphabets = reducedAlphabet<States>();


public:

    explicit MCDiscretizedScalesClassifier( Order mxOrder )
            : _discretizedAAScales( selectedIndices , States ) ,
              _modelTrainer( MG::template create<RMC>( mxOrder ))
    {}

    void runTraining( const std::map<std::string_view , std::vector<std::string >> &trainingClusters )
    {
        _discretizedAAScales.runClustering();
        auto transformedSequences = _transformSequences( trainingClusters , _discretizedAAScales );
        _backbones = MCModel::train( transformedSequences , _modelTrainer );
        _background = MCModel::backgroundProfiles( transformedSequences , _modelTrainer );
    }

    virtual ~MCDiscretizedScalesClassifier() = default;

protected:

    static inline char nearestNeighborInterpolateAminoAcids_OU( std::string_view sequence , const size_t index )
    {
        assert( sequence.at( index ) == 'O' || sequence.at( index ) == 'U' );
        auto aaItForward = std::find_if( sequence.cbegin() + index , sequence.cend() , [&]( char aa )
        {
            return aa != 'O' && aa != 'U';
        } );
        if ( aaItForward != sequence.cend())
        {
            return *aaItForward;
        } else
        {
            auto aaItBackward = std::find_if( sequence.crbegin() + (sequence.length() - index) , sequence.crend() ,
                                              [&]( char aa )
                                              {
                                                  return aa != 'O' && aa != 'U';
                                              } );
            if ( aaItBackward != sequence.crend())
            {
                return *aaItBackward;
            } else
            {
                return sequence.at( index );
            }
        }
    }

    static std::string _transformSequence( std::string_view sequence ,
                                           const aaindex::AAIndexClustering &discretizedAAScales )
    {
        std::string transformed;
        transformed.reserve( sequence.size());
        for ( size_t idx = 0; idx < sequence.length(); ++idx )
        {
            auto interpolate = [&]()
            {
                return nearestNeighborInterpolateAminoAcids_OU( sequence , idx );
            };
            auto cluster = [&]( char aa )
            {
                return discretizedAAScales.getCluster( aa );
            };
            auto aa = sequence.at( idx );
            if ( auto clusterOpt = cluster( aa ); clusterOpt )
                transformed.push_back( alphabets.at( clusterOpt.value()));
            else transformed.push_back( alphabets.at( cluster( interpolate()).value()));
        }

        return transformed;
    }

    static std::vector<std::string> _transformSequences(
            const std::vector<std::string> &sequences ,
            const aaindex::AAIndexClustering &discretizedAAScales )
    {
        std::vector<std::string> transformed;
        transformed.reserve( sequences.size());
        for ( auto &sequence : sequences )
            transformed.emplace_back( _transformSequence( sequence , discretizedAAScales ));
        return transformed;
    }

    static std::map<std::string_view , std::vector<std::string >> _transformSequences(
            const std::map<std::string_view , std::vector<std::string >> &labeledSequences ,
            const aaindex::AAIndexClustering &discretizedAAScales )
    {
        std::map<std::string_view , std::vector<std::string >> transformed;
        for ( auto&[label , sequences] : labeledSequences )
            transformed.emplace( label , _transformSequences( sequences , discretizedAAScales ));
        return transformed;
    }

    bool _validTraining() const override
    {
        return _backbones.size() == _background.size();
    }

    ScoredLabels _predict( std::string_view sequence ) const override
    {
        std::map<std::string_view , double> propensitites;

        auto tSequence = _transformSequence( sequence , _discretizedAAScales );

        for ( auto&[label , backbone] :_backbones )
        {
            auto &bg = _background.at( label );
            double logOdd = backbone->propensity( tSequence ) - bg->propensity( tSequence );
            propensitites[label] = logOdd;
        }

        propensitites = minmaxNormalize( std::move( propensitites ));

        ScoredLabels matchSet( _backbones.size());
        for ( auto &[label , relativeAffinity] : propensitites )
            matchSet.emplace( label , relativeAffinity );

        return matchSet;
    }

protected:
    ModelGenerator <States> _modelTrainer;
    BackboneProfiles _backbones;
    BackboneProfiles _background;
    aaindex::AAIndexClustering _discretizedAAScales;
};

}


#endif //MARKOVIAN_FEATURES_MCDISCRETIZEDSCALESCLASSIFIER_HPP
