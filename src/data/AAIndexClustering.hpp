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

    static constexpr auto euclidean = Criteria<Euclidean>::template function<SampleType>;

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
                    auto point = getPoint( aa );
                    return _closestIdx( _centroids.value() , point );
                } ));

    }

    SampleType getPoint( char aa ) const
    {
        std::vector<double> point;
        for ( auto &index : _index )
        {
            point.push_back( index.normalizedIndex()( aa ));
        }
        return vector_to_cmatrix( point );
    }

    auto getCluster( char aa ) const
    {
        assert( _centroids.has_value() && _clusters.has_value());
        return _clusters->at( aa );
    }

protected:

    long _closestIdx( const std::vector<SampleType> &centroids , SampleType point )
    {
        auto minIt = std::min_element(
                centroids.cbegin() , centroids.cend() ,
                [&]( SampleType x , SampleType y )
                {
                    return euclidean( x , point ) < euclidean( y , point );
                } );
        assert( minIt != centroids.cend());
        return std::distance( centroids.cbegin() , minIt );
    }

    std::vector<SampleType> _samples() const
    {
        std::vector<SampleType> samples;
        for ( auto aa : AMINO_ACIDS20 )
            samples.emplace_back(  getPoint( aa ));
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
