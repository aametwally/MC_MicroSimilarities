//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
#define MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP

#include <experimental/filesystem>

#include "common.hpp"
#include "aminoacids_grouping.hpp"

template<typename T>
class SequenceEntry
{
public:
    virtual size_t sequenceLength() const = 0;

    virtual const std::string &getSequence() const = 0;

    static std::map<std::string, std::vector<std::string> >
    groupSequencesByLabels( std::vector<T> &&entries )
    {
        std::map<std::string, std::vector<std::string >> clusters;

        for (auto &entry : entries)
            clusters[entry.getLabel()].emplace_back( entry.getSequence());

        return clusters;
    }

    static std::map<std::string, std::vector<T> >
    groupEntriesByLabels( std::vector<T> &&entries )
    {
        std::map<std::string, std::vector<T >> clusters;

        for (auto &&entry : entries)
            clusters[entry.getLabel()].emplace_back( std::move( entry ));

        return clusters;
    }

    static std::pair<std::vector<T>, std::vector<T >>
    separationExcludingClustersWithLowSequentialData(
            std::vector<T> &&entries,
            double percentage = 0.1f,
            double threshold = 10.f )
    {

        std::vector<T> subset, rest;
        auto clusterSize = []( const std::vector<T> &cluster ) {
            size_t count = 0;
            for (auto &s : cluster)
                count += s.sequenceLength();
            return count;
        };

        auto clusters = groupEntriesByLabels( std::move( entries ));

        size_t populationSequenceLength = 0;
        for (const auto &[clusterId, cluster] : clusters)
            populationSequenceLength += clusterSize( cluster );

        auto averageClusterSequenceSize = populationSequenceLength / clusters.size();

        for (const auto &[clusterId, cluster] : clusters)
            if ( clusterSize( cluster ) >= averageClusterSequenceSize * threshold )
            {
                const auto[subset_, rest_] = subsetRandomSeparation( cluster, percentage );
                subset.insert( subset.end(), subset_.cbegin(), subset_.cend());
                rest.insert( rest.end(), rest_.cbegin(), rest_.cend());
            }

        return std::make_pair( subset, rest );
    }

    static std::pair<std::vector<T>, std::vector<T >>
    separationExcludingClustersWithFewMembers(
            const std::vector<T> &entries,
            double percentage = 0.1f,
            double threshold = 5.f )
    {
        std::vector<T> subset, rest;
        auto clusters = groupEntriesByLabels( entries );

        auto averageClusterMembers = entries.size() / clusters.size();

        for (const auto &[clusterId, cluster] : clusters)
            if ( cluster.size() >= averageClusterMembers * threshold )
            {
                const auto[subset_, rest_] = subsetRandomSeparation( cluster, percentage );
                subset.insert( subset.end(), subset_.cbegin(), subset_.cend());
                rest.insert( rest.end(), rest_.cbegin(), rest_.cend());
            }

        return std::make_pair( subset, rest );
    }

    template<typename AAGrouping>
    static std::string reduceAlphabets( const std::string &sequence )
    {
        constexpr auto StatesN = AAGrouping::StatesN;
        constexpr auto Grouping = AAGrouping::Grouping;
        constexpr auto newAlphabet = reducedAlphabet<StatesN>();
        constexpr auto newAlphabetIds = reducedAlphabetIds( Grouping );

        std::string reducedSequence;
        reducedSequence.reserve( sequence.size());
        for (auto a : sequence)
            reducedSequence.push_back( newAlphabet.at( newAlphabetIds.at( a )));

        assert( reducedSequence.find( '\0' ) == std::string::npos );
        return reducedSequence;
    }

    template<typename AAGrouping, typename Entries>
    static std::vector<T>
    reducedAlphabetEntries( Entries &&unirefEntries )
    {
        std::vector<T> reducedEntries = std::forward<Entries>(unirefEntries);
        for ( T &ui : reducedEntries)
        {
            ui.setSequence( reduceAlphabets<AAGrouping>( ui.getSequence()));
        }
        return reducedEntries;
    }

};


#endif //MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
