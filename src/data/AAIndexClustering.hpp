//
// Created by asem on 01/01/19.
//

#ifndef MARKOVIAN_FEATURES_AAINDEXCLUSTERING_HPP
#define MARKOVIAN_FEATURES_AAINDEXCLUSTERING_HPP

#include "dlib_utilities.hpp"
#include <dlib/svm.h>

#include "SimilarityMetrics.hpp"
#include "LabeledEntry.hpp"
#include "AAIndexDBGET.hpp"
#include "common.hpp"
#include "LUT.hpp"

namespace aaindex
{

class AAIndexClustering
{
    using SampleType = dlib::matrix<double , 0 , 0>;

public:
    explicit AAIndexClustering( const std::vector<AAIndex1> &index ,
                                size_t nClusters )
            : _index( index ) ,
              _dims( index.size()) ,
              _nClusters( nClusters ) {}

    virtual void runClustering()
    {

        auto samplesVector = _samples();

        {
            std::vector<SampleType> centroids;

            dlib::pick_initial_centers( _nClusters , centroids , samplesVector );
            dlib::find_clusters_using_kmeans( samplesVector , centroids );

            _centroids.emplace( std::move( centroids ));
        }

        _clusters.emplace( LUT<char , long>::makeLUT(
                [this]( char aa ) -> long
                {
                    if ( auto point = getPoint( aa ); point )
                        return _closestIdx( _centroids.value() , point.value());
                    else return -1;
                } ));
    }

    std::optional<SampleType> getPoint( char aa ) const
    {
        using namespace dlib_utilities;
        std::vector<double> point;
        for ( auto &index : _index )
        {
            if ( auto component = index.normalizedIndex( aa ); component )
                point.push_back( component.value());
            else return std::nullopt;
        }
        return vectorToColumnMatrixLike( std::move( point ));
    }

    std::optional<size_t> getCluster( char aa ) const
    {
        assert( _centroids.has_value() && _clusters.has_value());
        if ( auto cluster = _clusters->at( aa ); cluster >= 0 )
            return size_t( cluster );
        else return std::nullopt;
    }

protected:

    long _closestIdx( const std::vector<SampleType> &centroids , const SampleType &point )
    {
        static auto euclidean = Euclidean::template similarityFunctor<SampleType>();

        auto minIt = std::min_element(
                centroids.cbegin() , centroids.cend() ,
                [&]( SampleType x , SampleType y )
                {
                    return euclidean( x , point ) < euclidean( y , point );
                } );
        return std::distance( centroids.cbegin() , minIt );
    }

    std::vector<SampleType> _samples() const
    {
        std::vector<SampleType> samples;
        for ( auto aa : AMINO_ACIDS20 )
        {
            auto point = getPoint( aa );
            assert( point.has_value());
            samples.emplace_back( getPoint( aa ).value());
        }
        return samples;
    }


private:
    const std::vector<AAIndex1> _index;
    const size_t _nClusters;
    const size_t _dims;

    std::optional<std::vector<SampleType>> _centroids;
    std::optional<LUT<char , long>> _clusters;
};

}


#endif //MARKOVIAN_FEATURES_AAINDEXCLUSTERING_HPP
