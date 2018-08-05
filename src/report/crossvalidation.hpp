//
// Created by asem on 05/08/18.
//

#ifndef MARKOVIAN_FEATURES_CROSSVALIDATION_HPP
#define MARKOVIAN_FEATURES_CROSSVALIDATION_HPP

#include <chrono>

#include "common.hpp"

template<typename T>
std::vector<std::vector<T >>
kFoldSplit( std::vector<T> input, size_t k )
{
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle( input.begin(), input.end(), std::default_random_engine( seed ));
    std::vector<std::vector<T >> folds( k, std::vector<T>());
    for (auto i = 0; i < input.size(); ++i)
        folds.at( i * k / input.size()).push_back( input.at( i ));
    return folds;
}


template< typename T, typename Label = std::string >
std::vector<std::vector<std::pair<Label, T >>>
kFoldStratifiedSplit( const std::map<Label, std::vector<T> > &input, size_t k )
{
    std::vector<std::vector<std::pair<Label, T >>> folds( k );
    for (auto &[label, v] : input)
    {
        auto sFolds = kFoldSplit( v, k );
        for (auto i = 0; i < sFolds.size(); ++i)
        {
            auto &foldK = folds.at( i );
            auto &sFoldK = sFolds.at( i );
            std::transform( std::begin( sFoldK ), std::end( sFoldK ),
                            std::back_inserter( foldK ), [&]( const T &item ) {
                        return std::make_pair( label, item );
                    } );
        }
    }

    return folds;
}

template<typename T, typename Label = std::string >
std::map<Label, std::vector<T >>
joinFoldsExceptK( const std::vector<std::vector<std::pair<Label, T >>> &input, size_t k )
{
    assert( k < input.size());

    std::map<Label, std::vector<T >> joined;
    auto mapInserter = [&]( const std::vector<std::pair<Label, T >> &vec ) {
        for (auto &p : vec)
            joined[ p.first ].push_back( p.second );
    };

    for (auto i = 0; i < input.size(); ++i)
    {
        if ( i == k ) continue;
        else mapInserter( input.at( i ));
    }
    return joined;
};

#endif //MARKOVIAN_FEATURES_CROSSVALIDATION_HPP
