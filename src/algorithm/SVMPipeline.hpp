//
// Created by asem on 13/08/18.
//

#ifndef MARKOVIAN_FEATURES_SVMPIPELINE_HPP
#define MARKOVIAN_FEATURES_SVMPIPELINE_HPP

#include "common.hpp"
#include "VariantGenerator.hpp"
#include "LabeledEntry.hpp"
#include "Timers.hpp"
#include "ConfusionMatrix.hpp"
#include "CrossValidationStatistics.hpp"
#include "crossvalidation.hpp"
#include "SVMMarkovianModel.hpp"

template<typename Grouping>
class SVMPipeline
{
    using SVMModel = SVMMarkovianModel<Grouping>;

    using Order = typename SVMModel::Order;
public:
    static std::vector<LabeledEntry>
    reducedAlphabetEntries( std::vector<LabeledEntry> &&entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( entries );
    }


    void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries,
                                 Order order,
                                 size_t k )
    {
        using Folds = std::vector<std::vector<std::pair<std::string, std::string >>>;

        fmt::print( "[All Sequences:{}]\n", entries.size());

        std::set<std::string> labels;
        for (const auto &entry : entries)
            labels.insert( entry.getLabel());

        auto entriesReduced = reducedAlphabetEntries( std::move( entries ));

        auto groupedEntries = LabeledEntry::groupSequencesByLabels( std::move( entriesReduced ));

        {
            auto _labels = keys( groupedEntries );
            for (auto &l : _labels) l = fmt::format( "{}({})", l, groupedEntries.at( l ).size());
            fmt::print( "[Clusters:{}][{}]\n",
                        groupedEntries.size(),
                        io::join( _labels, "|" ));
        }



        const Folds folds = kFoldStratifiedSplit( std::move( groupedEntries ), k );

        auto extractTest = []( const std::vector<std::pair<std::string, std::string >> &items ) {
            std::vector<std::string> sequences, labels;
            for (const auto &item : items)
            {
                labels.push_back( item.first );
                sequences.push_back( item.second );
            }
            return std::make_pair( sequences, labels );
        };

        CrossValidationStatistics validation( k, labels );

        for (auto i = 0; i < k; ++i)
        {
            SVMModel svmModel( order );
            auto trainingClusters = joinFoldsExceptK( folds, i );
            auto[test, tLabels] = extractTest( folds.at( i ));
            svmModel.fit( trainingClusters );
            auto predicted = svmModel.predict( test  );

            for( auto ii = 0 ; ii < test.size() ; ++ii )
                validation.countInstance( i , predicted.at( ii ) , tLabels.at( ii ));
            validation.printReport(i);
        }
        validation.printReport();
    }
};



using SVMPipelineVariant = MakeVariantType<SVMPipeline,
        SupportedAAGrouping>;

SVMPipelineVariant getSVMPipeline( const std::string &groupingLabel )
{
    const AminoAcidGroupingEnum grouping = GroupingLabels.at( groupingLabel );
    switch (grouping)
    {
        case AminoAcidGroupingEnum::NoGrouping20:
            return SVMPipeline<AAGrouping_NOGROUPING20>();
        case AminoAcidGroupingEnum::DIAMOND11 :
            return SVMPipeline<AAGrouping_DIAMOND11>();
        case AminoAcidGroupingEnum::OFER8 :
            return SVMPipeline<AAGrouping_OFER8>();
        case AminoAcidGroupingEnum::OFER15 :
            return SVMPipeline<AAGrouping_OFER15>();
    }
};

#endif //MARKOVIAN_FEATURES_SVMPIPELINE_HPP
