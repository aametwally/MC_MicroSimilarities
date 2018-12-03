//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
#define MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP

#include <experimental/filesystem>
#include "AAGrouping.hpp"
#include "AAIndexDBGET.hpp"
#include "common.hpp"


template < typename T >
class SequenceEntry
{
public:
    virtual size_t sequenceLength() const = 0;

    virtual const std::string &getSequence() const = 0;

    virtual ~SequenceEntry() = default;

    static std::map<std::string , std::vector<std::string> >
    groupSequencesByLabels( std::vector<T> &&entries )
    {
        std::map<std::string , std::vector<std::string >> clusters;

        for ( auto &entry : entries )
            clusters[entry.getLabel()].emplace_back( entry.getSequence());

        return clusters;
    }

    static std::map<std::string , std::vector<T> >
    groupEntriesByLabels( std::vector<T> &&entries )
    {
        std::map<std::string , std::vector<T >> clusters;

        for ( auto &&entry : entries )
            clusters[entry.getLabel()].emplace_back( std::move( entry ));

        return clusters;
    }

    static std::pair<std::vector<T> , std::vector<T >>
    separationExcludingClustersWithLowSequentialData(
            std::vector<T> &&entries ,
            double percentage = 0.1f ,
            double threshold = 10.f )
    {

        std::vector<T> subset , rest;
        auto clusterSize = []( const std::vector<T> &cluster )
        {
            size_t count = 0;
            for ( auto &s : cluster )
                count += s.sequenceLength();
            return count;
        };

        auto clusters = groupEntriesByLabels( std::move( entries ));

        size_t populationSequenceLength = 0;
        for ( const auto &[clusterId , cluster] : clusters )
            populationSequenceLength += clusterSize( cluster );

        auto averageClusterSequenceSize = populationSequenceLength / clusters.size();

        for ( const auto &[clusterId , cluster] : clusters )
            if ( clusterSize( cluster ) >= averageClusterSequenceSize * threshold )
            {
                const auto[subset_ , rest_] = subsetRandomSeparation( cluster , percentage * cluster.size());
                subset.insert( subset.end() , subset_.cbegin() , subset_.cend());
                rest.insert( rest.end() , rest_.cbegin() , rest_.cend());
            }

        return std::make_pair( subset , rest );
    }

    static std::pair<std::vector<T> , std::vector<T >>
    separationExcludingClustersWithFewMembers(
            const std::vector<T> &entries ,
            double percentage = 0.1f ,
            double threshold = 5.f )
    {
        std::vector<T> subset , rest;
        auto clusters = groupEntriesByLabels( entries );

        auto averageClusterMembers = entries.size() / clusters.size();

        for ( const auto &[clusterId , cluster] : clusters )
            if ( cluster.size() >= averageClusterMembers * threshold )
            {
                const auto[subset_ , rest_] = subsetRandomSeparation( cluster , percentage * cluster.size());
                subset.insert( subset.end() , subset_.cbegin() , subset_.cend());
                rest.insert( rest.end() , rest_.cbegin() , rest_.cend());
            }

        return std::make_pair( subset , rest );
    }

    static inline bool isPolymorphicAA( char aa )
    {
        return POLYMORPHIC_AA.find( aa ) != POLYMORPHIC_AA.cend();
    }

//    static inline bool isPolymorphicSequence( std::string_view sequence )
//    {
//        return std::any_of( sequence.cbegin() , sequence.cend() , isPolymorphicAA );
//    }

    static inline bool isPolymorphicReducedAA( char aa )
    {
        return POLYMORPHIC_AA_DECODE.find( aa ) != POLYMORPHIC_AA_DECODE.cend();
    }

    template < typename AAGrouping >
    static inline bool isPolymorphicReducedSequence( std::string_view sequence )
    {
        assert( isReducedSequence<AAGrouping>( sequence ));
        return isReducedSequence<AAGrouping>( sequence ) &&
               std::any_of( sequence.cbegin() , sequence.cend() , isPolymorphicReducedAA );
    }

    template < typename AAGrouping , typename Sequence >
    static inline bool isReducedSequences( const std::vector<Sequence> &sequences )
    {
        return std::all_of( sequences.cbegin() , sequences.cend() , isReducedSequence < AAGrouping > );
    }

    template < typename AAGrouping >
    static inline std::string generateReducedAAMutations( char aa )
    {
        assert( isPolymorphicReducedAA( aa ));
        char polymorphicAA = POLYMORPHIC_AA_DECODE.at( aa );
        const auto &mutations = POLYMORPHIC_AA.at( polymorphicAA );
        std::string reducedMutations;
        reducedMutations.reserve( mutations.size());
        std::transform( mutations.cbegin() , mutations.cend() ,
                        std::back_inserter( reducedMutations ) , reduceAlphabet < AAGrouping > );
        return reducedMutations;
    }

    template < typename AAGrouping >
    static inline std::vector<std::string>
    mutateFirstReducedPolymorphicAA( std::string_view sequence )
    {
        assert( isPolymorphicReducedSequence<AAGrouping>( sequence ));
        auto polymorphicIt = std::find_if( sequence.cbegin() , sequence.cend() , isPolymorphicReducedAA );
        if ( polymorphicIt == sequence.cend())
        {
            return {};
        } else
        {
            auto position = std::distance( sequence.cbegin() , polymorphicIt );
            char polymorphicAA = POLYMORPHIC_AA_DECODE.at( *polymorphicIt );
            const auto &mutations = POLYMORPHIC_AA.at( polymorphicAA );
            std::vector<std::string> mutatedSequences;
            mutatedSequences.reserve( mutations.size());
            for ( auto mutation : mutations )
            {
                char reducedMutation = reduceAlphabet<AAGrouping>( mutation );
                std::string mutatedSequence = std::string( sequence );
                mutatedSequences.emplace_back( sequence );
                mutatedSequences.back()[position] = reducedMutation;
            }
            return mutatedSequences;
        }
    }

    template < typename AAGrouping >
    static inline std::vector<std::string>
    generateReducedPolymorphicSequenceOutcome( std::string_view sequence )
    {
        assert( isReducedSequence<AAGrouping>( sequence ));
        if ( !isPolymorphicReducedSequence<AAGrouping>( sequence ))
            return {std::string( sequence )};
        else
        {
            std::vector<std::string> nonPolymorphicSequences;
            std::list<std::string> polymorphicSequences = {std::string( sequence )};

            while ( !polymorphicSequences.empty())
            {
                std::string current( std::move( polymorphicSequences.front()));
                polymorphicSequences.pop_front();
                auto mutatedSequences = mutateFirstReducedPolymorphicAA<AAGrouping>( current );
                if ( isPolymorphicReducedSequence<AAGrouping>( mutatedSequences.front()))
                {
                    polymorphicSequences.insert( polymorphicSequences.end() ,
                                                 std::make_move_iterator( mutatedSequences.begin()) ,
                                                 std::make_move_iterator( mutatedSequences.end()));
                } else
                {
                    nonPolymorphicSequences.reserve( nonPolymorphicSequences.size() + mutatedSequences.size());
                    nonPolymorphicSequences.insert( nonPolymorphicSequences.end() ,
                                                    std::make_move_iterator( mutatedSequences.begin()) ,
                                                    std::make_move_iterator( mutatedSequences.end()));
                }
            }
            return nonPolymorphicSequences;
        }
    }

    template < typename AAGrouping >
    static inline bool isReducedAA( char aa )
    {
        static constexpr auto ReducedAlphabet = reducedAlphabet<AAGrouping::StatesN>();
        return std::find( ReducedAlphabet.cbegin() , ReducedAlphabet.cend() , aa ) != ReducedAlphabet.cend()
               || POLYMORPHIC_AA_DECODE.find( aa ) != POLYMORPHIC_AA_DECODE.cend();
    }

    template < typename AAGrouping >
    static inline bool isReducedSequence( std::string_view sequence )
    {
        return std::all_of( sequence.cbegin() , sequence.cend() , isReducedAA<AAGrouping> );
    }

    template < typename AAGrouping >
    static char reduceAlphabet( char aa )
    {
        static constexpr auto StatesN = AAGrouping::StatesN;
        static constexpr auto Grouping = AAGrouping::Grouping;
        static constexpr auto newAlphabet = reducedAlphabet<StatesN>();
        static constexpr auto newAlphabetIds = reducedAlphabetIds( Grouping );
        return newAlphabet.at( newAlphabetIds.at( aa ));
    }

    template < typename AAGrouping >
    static std::string reduceAlphabets( std::string_view peptide )
    {

        std::string reducedPeptide;
        reducedPeptide.reserve( peptide.size());
        for ( auto a : peptide )
        {
            if ( isPolymorphicAA( a ))
            {
                reducedPeptide.push_back( POLYMORPHIC_AA_ENCODE.at( a ));
            } else
            {
                reducedPeptide.push_back( reduceAlphabet<AAGrouping>( a ));
            }
        }

        assert( isReducedSequence<AAGrouping>( reducedPeptide ));
        return reducedPeptide;
    }

    template < typename AAGrouping , typename Entries >
    static std::vector<T>
    reducedAlphabetEntries( Entries &&unirefEntries )
    {
        std::vector<T> reducedEntries = std::forward<Entries>( unirefEntries );
        for ( T &ui : reducedEntries )
        {
            ui.setSequence( reduceAlphabets<AAGrouping>( ui.getSequence()));
        }
        return reducedEntries;
    }

};


#endif //MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
