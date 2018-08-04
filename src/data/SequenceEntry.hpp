//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
#define MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP

#include "common.hpp"
#include "aminoacids_grouping.hpp"

template<typename T>
class SequenceEntry
{
public:
    virtual size_t sequenceLength() const = 0;

    virtual std::string clusterNameUniRef() const = 0;

    virtual const std::string &getSequence() const = 0;

    static std::map<std::string, std::vector<std::string> >
    groupSequencesByUniRefClusters(const std::vector<T> &entries)
    {
        std::map<std::string, std::vector<std::string >> clusters;

        for (const auto &entry : entries)
            clusters[entry.clusterNameUniRef()].emplace_back(entry.getSequence());

        return clusters;
    }

    static std::map<std::string, std::vector<T> >
    groupEntriesByUniRefClusters(const std::vector<T> &entries)
    {
        std::map<std::string, std::vector<T >> clusters;

        for (const auto &entry : entries)
            clusters[entry.clusterNameUniRef()].emplace_back(entry);

        return clusters;
    }

    static std::pair<std::vector<T>, std::vector<T >>
    separationExcludingClustersWithLowSequentialData(
            const std::vector<T> &entries,
            float percentage = 0.1f,
            float threshold = 10.f)
    {

        std::vector<T> subset, rest;
        auto clusterSize = [](const std::vector<T> &cluster) {
            size_t count = 0;
            for (auto &s : cluster)
                count += s.sequenceLength();
            return count;
        };

        auto clusters = groupEntriesByUniRefClusters(entries);

        size_t populationSequenceLength = 0;
        for (const auto &[clusterId, cluster] : clusters)
            populationSequenceLength += clusterSize(cluster);

        auto averageClusterSequenceSize = populationSequenceLength / clusters.size();

        for (const auto &[clusterId, cluster] : clusters)
            if (clusterSize(cluster) >= averageClusterSequenceSize * threshold) {
                const auto[subset_, rest_] = subsetRandomSeparation(cluster, percentage);
                subset.insert(subset.end(), subset_.cbegin(), subset_.cend());
                rest.insert(rest.end(), rest_.cbegin(), rest_.cend());
            }

        return std::make_pair(subset, rest);
    }

    static std::pair<std::vector<T>, std::vector<T >>
    separationExcludingClustersWithFewMembers(
            const std::vector<T> &entries,
            float percentage = 0.1f,
            float threshold = 5.f)
    {
        std::vector<T> subset, rest;
        auto clusters = groupEntriesByUniRefClusters(entries);

        auto averageClusterMembers = entries.size() / clusters.size();

        for (const auto &[clusterId, cluster] : clusters)
            if (cluster.size() >= averageClusterMembers * threshold) {
                const auto[subset_, rest_] = subsetRandomSeparation(cluster, percentage);
                subset.insert(subset.end(), subset_.cbegin(), subset_.cend());
                rest.insert(rest.end(), rest_.cbegin(), rest_.cend());
            }

        return std::make_pair(subset, rest);
    }

    template< size_t N , const std::array< const char *, N > &Grouping >
    static std::string reduceAlphabets(const std::string &sequence)
    {
        constexpr auto newAlphabet = reducedAlphabet<N>();
        constexpr auto newAlphabetIds = reducedAlphabetIds( Grouping );

        std::string reducedSequence;
        reducedSequence.reserve(sequence.size());
        for (auto a : sequence)
            reducedSequence.push_back(newAlphabet.at(newAlphabetIds.at(a)));

        assert(reducedSequence.find('\0') == std::string::npos);
        return reducedSequence;
    }

    template< size_t N , const std::array< const char *, N > &Grouping >
    static std::vector<T>
    reducedAlphabetEntries(const std::vector<T> &unirefEntries)
    {
        std::vector<T> unirefReducedEntries;
        for (const T &ui : unirefEntries) {
            auto reduced = ui;
            reduced.setSequence(reduceAlphabets<N,Grouping>(ui.getSequence()));
            unirefReducedEntries.emplace_back(reduced);
        }
        return unirefReducedEntries;
    }

};


#endif //MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
