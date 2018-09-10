//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_ZYPIPELINE_HPP
#define MARKOVIAN_FEATURES_ZYPIPELINE_HPP
#include "common.hpp"
#include "VariantGenerator.hpp"
#include "LabeledEntry.hpp"
#include "ConfusionMatrix.hpp"
#include "CrossValidationStatistics.hpp"
#include "crossvalidation.hpp"
#include "ZYMC.hpp"
#include "similarities.hpp"

template< typename Grouping = AAGrouping_NOGROUPING20 >
class ZYPipeline
{
public:

private:
    using PriorityQueue = typename MatchSet<Score>::Queue;
    using LeaderBoard = ClassificationCandidates<Score>;

    using ZYMC = MC::ZYMC<Grouping>;
    using BackboneProfiles =  typename ZYMC::BackboneProfiles;

public:

    template<typename Entries>
    static std::vector <LabeledEntry>
    reducedAlphabetEntries( Entries &&entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( std::forward<Entries>( entries ));
    }

    static std::vector <LeaderBoard>
    classify_VALIDATION(
            const std::vector <std::string> &queries,
            const std::vector <std::string> &trueLabels,
            BackboneProfiles &&targets )
    {
        assert( queries.size() == trueLabels.size());

        auto results = predict( queries, targets );
        assert( results.size() == queries.size());
        std::vector <LeaderBoard> classifications;
        for (auto i = 0; i < queries.size(); ++i)
        {
            classifications.emplace_back( trueLabels.at( i ),
                                          results.at( i ));
        }
        return classifications;
    }

    static std::vector <PriorityQueue>
    predict( const std::vector <std::string> &queries,
             const BackboneProfiles &targets  )
    {
        std::vector< PriorityQueue  > rankedPredictions;
        for( auto &query : queries )
        {
            PriorityQueue matchSet( targets.size());
            for( auto&[label,backbone] :targets )
            {
                matchSet.emplace( label , backbone.propensity( query ));
            }
            rankedPredictions.emplace_back( std::move( matchSet ));
        }
        return rankedPredictions;
    }

    BackboneProfiles train( const std::map< std::string , std::vector< std::string > > &trainingData, Order order )
    {
        BackboneProfiles profiles;
        for( auto &[label,sequences] : trainingData )
            profiles.emplace( label , ZYMC( sequences , order ));
        return profiles;
    }

    void runPipeline_VALIDATION( std::vector <LabeledEntry> &&entries, Order order, size_t k )
    {
        std::set <std::string> labels;
        for (const auto &entry : entries)
            labels.insert( entry.getLabel());

        using Folds = std::vector <std::vector<std::pair < std::string, std::string >>>;
        auto groupedEntries = LabeledEntry::groupSequencesByLabels( reducedAlphabetEntries( std::move( entries )));
        const Folds folds = kFoldStratifiedSplit( std::move( groupedEntries ), k );

        auto extractTest = []( const std::vector <std::pair<std::string, std::string >> &items ) {
            std::vector <std::string> sequences, labels;
            for (const auto item : items)
            {
                labels.push_back( item.first );
                sequences.push_back( item.second );
            }
            return std::make_pair( sequences, labels );
        };

        CrossValidationStatistics validation( k, labels );
        std::unordered_map<long, size_t> histogram;

        for (auto i = 0; i < k; ++i)
        {
            auto trainingClusters = joinFoldsExceptK( folds, i );
            auto[test, tLabels] = extractTest( folds.at( i ));
            auto trainedProfiles = train( trainingClusters, order );
            auto classificationResults = classify_VALIDATION( test, tLabels, std::move( trainedProfiles ));

            for (const auto &classification : classificationResults)
            {
                if ( auto prediction = classification.bestMatch();prediction )
                {
                    ++histogram[classification.trueClusterRank()];
                    validation.countInstance( i, prediction.value(), classification.trueCluster());
                } else
                {
                    ++histogram[-1];
                    validation.countInstance( i, "unclassified", classification.trueCluster());
                }
            }
        }

        validation.printReport();

        fmt::print( "True Classification Histogram:\n" );

        for (auto &[k, v] : histogram)
        {
            if ( k == -1 )
                fmt::print( "[{}:{}]", "Unclassified", v );
            else
                fmt::print( "[{}:{}]", fmt::format( "Rank{}", k ), v );
        }
        fmt::print( "\n" );
    }
};


using ZYipelineVariant = MakeVariantType<ZYPipeline,
        SupportedAAGrouping>;

ZYipelineVariant getZYPipeline( const std::string &groupingLabel )
{
    const AminoAcidGroupingEnum grouping = GroupingLabels.at( groupingLabel );
    switch (grouping)
    {
        case AminoAcidGroupingEnum::NoGrouping20:
            return ZYPipeline<AAGrouping_NOGROUPING20>();
        case AminoAcidGroupingEnum::DIAMOND11 :
            return ZYPipeline<AAGrouping_DIAMOND11>();
        case AminoAcidGroupingEnum::OLFER8 :
            return ZYPipeline<AAGrouping_OLFER8>();
        case AminoAcidGroupingEnum::OLFER15 :
            return ZYPipeline<AAGrouping_OLFER15>();
    }
};

#endif //MARKOVIAN_FEATURES_ZYPIPELINE_HPP
