//
// Created by asem on 12/09/18.
//

#ifndef MARKOVIAN_FEATURES_KNNMODEL_HPP
#define MARKOVIAN_FEATURES_KNNMODEL_HPP

#include "common.hpp"
#include "SimilarityMetrics.hpp"

template<typename Distance = Euclidean>
class KNNModel
{
public:
    explicit KNNModel( size_t k = 3 ) : _k( k )
    {}

    virtual ~KNNModel() = default;

    void setK( size_t k )
    {
        _k = k;
    }

    size_t getK() const
    {
        return _k;
    }

    void fit(
            const std::vector<std::string_view> &labels,
            std::vector<std::vector<double >> &&features
    )
    {
        assert( labels.size() == features.size());
        const size_t featuresSpace = features.front().size();
        _population.clear();
        _indices.clear();
        assert( std::all_of( features.begin(), features.end(),
                             [=]( auto &v ) { return v.size() == featuresSpace; } ));

        for (size_t i = 0; i < labels.size(); ++i)
        {
            _population[labels.at( i )].emplace_back( std::move( features.at( i )));
            _indices[i] = _population.find( labels.at( i ))->first;
        }

        size_t id = 0;
        for (const auto &[clusterLabel, cluster] : _population)
            for (const auto &point : cluster)
            {
                _indices[id] = clusterLabel;
                ++id;
            }
    }

    ScoredLabels predict( const std::vector<double> &f ) const
    {
        static const auto distanceFunctor = Distance::template similarityFunctor<std::vector<double>>();

        PenalizedIndices topK( _k );
        size_t id = 0;
        for (const auto &[clusterLabel, cluster] : _population)
            for (const auto &point : cluster)
            {
                topK.emplace( id, distanceFunctor( f, point ));
                ++id;
            }
        std::map<std::string_view, double> voter;

        topK.forTopK( _k, [&](
                const auto &candidate,
                size_t index
        ) {
            size_t label = candidate.label();
            voter[_indices.at( label )] += 1;
        } );
        double max = 0;
        for (auto &[idx, label] : _indices)
            max = std::max( max, voter[label] += 0 );


        ScoredLabels vPQ( _population.size());
        for (const auto &[label, votes]: voter)
            vPQ.emplace( label, votes / max );
        return vPQ;
    }

private:
    size_t _k;
    std::map<std::string_view, std::vector<std::vector<double> >> _population;
    std::unordered_map<size_t, std::string_view> _indices;
};


#endif //MARKOVIAN_FEATURES_KNNMODEL_HPP
