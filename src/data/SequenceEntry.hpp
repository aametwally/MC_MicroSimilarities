//
// Created by asem on 31/07/18.
//

#ifndef MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
#define MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP

#include <experimental/filesystem>
#include "AAGrouping.hpp"
#include "common.hpp"
#include "LUT.hpp"

template<typename T>
class SequenceEntry
{
public:
    virtual size_t length() const = 0;

    virtual std::string_view sequence() const = 0;

    virtual ~SequenceEntry() = default;

    static std::map<std::string, std::vector<std::string> >
    groupSequencesByLabels( std::vector<T> &&entries )
    {
        std::map<std::string, std::vector<std::string >> clusters;

        for (auto &entry : entries)
            clusters[std::string( entry.label())].emplace_back( entry.sequence());

        return clusters;
    }

    static std::map<std::string, std::vector<T> >
    groupEntriesByLabels( std::vector<T> &&entries )
    {
        std::map<std::string, std::vector<T >> clusters;

        for (auto &&entry : entries)
            clusters[std::string( entry.label())].emplace_back( std::move( entry ));

        return clusters;
    }

    template<typename InputSequence, typename MapType>
    static auto
    groupAveragedValue(
            const MapType &groups,
            std::function<double(
                    std::string_view,
                    const InputSequence &
            )> measuringFunction
    )
    {
        using K = typename std::remove_reference_t<MapType>::key_type;
        using V = typename std::remove_reference_t<MapType>::value_type;

        std::map<K, double> values;
        for (const auto &[l, data] : groups)
        {
            auto &groupValue = values[l];
            for (const auto &item : data)
            {
                groupValue += measuringFunction( l, item );
            }
            groupValue /= data.size();
        }
        return values;
    }


    static inline bool isPolymorphicAA( char aa )
    {
        static auto lut = LUT<char, bool>::makeLUT( []( char a ) {
            return POLYMORPHIC_AA.find( a ) != POLYMORPHIC_AA.cend();
        } );
        return lut.at( aa );
    }

    static inline bool isPolymorphicReducedAA( char aa )
    {
        static auto lut = LUT<char, bool>::makeLUT( []( char a ) {
            return POLYMORPHIC_AA_DECODE.find( a ) !=
                   POLYMORPHIC_AA_DECODE.cend();
        } );
        return lut.at( aa );
    }

    template<size_t States>
    static inline bool isPolymorphicReducedSequence( std::string_view sequence )
    {
        assert( isReducedSequence<States>( sequence ));
        return std::any_of( sequence.cbegin(), sequence.cend(), isPolymorphicReducedAA );
    }


    template<size_t States, typename Sequence>
    static inline bool isReducedSequences( const std::vector<Sequence> &sequences )
    {
        return std::all_of( sequences.cbegin(), sequences.cend(), isReducedSequence < States > );
    }

    template<typename AAGrouping>
    static inline const std::string &generateReducedAAMutations( char aa )
    {
        assert( isPolymorphicReducedAA( aa ));
        static const auto lut = LUT<char, std::string>::makeLUT( []( char a ) -> std::string {
            if ( auto polymorphicAAIt = POLYMORPHIC_AA_DECODE.find(
                        a ); polymorphicAAIt !=
                             POLYMORPHIC_AA_DECODE.cend())
            {
                char polymorphicAA = polymorphicAAIt->second;
                const auto &mutations = POLYMORPHIC_AA.at(
                        polymorphicAA );
                std::string reducedMutations;
                reducedMutations.reserve( mutations.size());
                std::transform( mutations.cbegin(),
                                mutations.cend(),
                                std::back_inserter(
                                        reducedMutations ),
                                reduceAlphabet <
                                AAGrouping > );
                return reducedMutations;
            } else return "";
        } );
        return lut.at( aa );
    }

    template<typename AAGrouping>
    static inline std::pair<std::vector<std::string>, size_t>
    mutateFirstReducedPolymorphicAA(
            std::string_view sequence,
            int64_t offset = 0
    )
    {
        static constexpr auto States = AAGrouping::StatesN;
        assert( isPolymorphicReducedSequence<States>( sequence ));
        auto polymorphicIt = std::find_if( sequence.cbegin() + offset, sequence.cend(), isPolymorphicReducedAA );
        offset = std::distance( sequence.cbegin(), polymorphicIt );
        if ( polymorphicIt == sequence.cend())
        {
            return std::make_pair( std::vector<std::string>{}, offset );
        } else
        {
            char polymorphicAA = POLYMORPHIC_AA_DECODE.at( *polymorphicIt );
            const auto &mutations = POLYMORPHIC_AA.at( polymorphicAA );
            std::vector<std::string> mutatedSequences;
            mutatedSequences.reserve( mutations.size());
            for (auto mutation : mutations)
            {
                char reducedMutation = reduceAlphabet<AAGrouping>( mutation );
                std::string mutatedSequence = std::string( sequence );
                mutatedSequences.emplace_back( sequence );
                mutatedSequences.back()[offset] = reducedMutation;
            }
            return std::make_pair( std::move( mutatedSequences ), offset );
        }
    }

    template<typename AAGrouping>
    static inline std::list<std::string>
    generateReducedPolymorphicSequenceOutcome( std::string_view sequence )
    {
        static constexpr auto States = AAGrouping::StatesN;
        assert( isReducedSequence<States>( sequence ));
        if ( !isPolymorphicReducedSequence<States>( sequence ))
            return {std::string( sequence )};
        else
        {
            std::list<std::string> nonPolymorphicSequences;
            std::list<std::string> polymorphicSequences = {std::string( sequence )};
            int64_t offset = 0;
            while (!polymorphicSequences.empty())
            {
                std::string current( std::move( polymorphicSequences.front()));
                polymorphicSequences.pop_front();
                auto[mutatedSequences, newOffset] = mutateFirstReducedPolymorphicAA<AAGrouping>( current, offset );
                offset = newOffset;
                if ( isPolymorphicReducedSequence<States>( mutatedSequences.front()))
                {
                    polymorphicSequences.insert( polymorphicSequences.end(),
                                                 std::make_move_iterator( mutatedSequences.begin()),
                                                 std::make_move_iterator( mutatedSequences.end()));
                } else
                {
                    nonPolymorphicSequences.insert( nonPolymorphicSequences.end(),
                                                    std::make_move_iterator( mutatedSequences.begin()),
                                                    std::make_move_iterator( mutatedSequences.end()));
                }
            }
            return nonPolymorphicSequences;
        }
    }

    template<size_t States>
    static inline bool isReducedAA( char aa )
    {
        static constexpr auto ReducedAlphabet = reducedAlphabet<States>();
        return std::find( ReducedAlphabet.cbegin(), ReducedAlphabet.cend(), aa ) != ReducedAlphabet.cend()
               || POLYMORPHIC_AA_DECODE.find( aa ) != POLYMORPHIC_AA_DECODE.cend();
    }

    template<size_t States>
    static inline bool isReducedSequence( std::string_view sequence )
    {
        return std::all_of( sequence.cbegin(), sequence.cend(), isReducedAA<States> );
    }

    template<typename AAGrouping>
    static char reduceAlphabet( char aa )
    {
        static constexpr auto StatesN = AAGrouping::StatesN;
        static constexpr auto Grouping = AAGrouping::Grouping;
        static constexpr auto newAlphabet = reducedAlphabet<StatesN>();
        static constexpr auto newAlphabetIds = reducedAlphabetIds( Grouping );

        if ( isPolymorphicAA( aa ))
        {
            return POLYMORPHIC_AA_ENCODE.at( aa );
        } else
        {
            return newAlphabet.at( newAlphabetIds.at( aa ));
        }
    }

    template<typename AAGrouping>
    static std::string reduceAlphabets( std::string_view peptide )
    {
        static constexpr auto States = AAGrouping::StatesN;
        std::string reducedPeptide;
        reducedPeptide.reserve( peptide.size());
        for (auto a : peptide)
            reducedPeptide.push_back( reduceAlphabet<AAGrouping>( a ));

        assert( isReducedSequence<States>( reducedPeptide ));
        return reducedPeptide;
    }

    template<typename AAGrouping, typename InputSequence>
    static std::vector<std::string>
    reducedAlphabetEntries( const std::vector<InputSequence> &entries )
    {
        std::vector<std::string> reduced;
        std::transform( entries.cbegin(), entries.cend(),
                        std::back_inserter( reduced ), reduceAlphabets<AAGrouping> );
        return reduced;
    }

    template<typename AAGrouping, typename Entries>
    static std::vector<T>
    reducedAlphabetEntries( Entries &&unirefEntries )
    {
        std::vector<T> reducedEntries = std::forward<Entries>( unirefEntries );
        for (T &ui : reducedEntries)
        {
            ui.setSequence( reduceAlphabets<AAGrouping>( ui.getSequence()));
        }
        return reducedEntries;
    }

    template<typename AAGrouping>
    static double polymorphicSummer(
            char polymorphicState,
            std::function<double( char )> fn,
            double acc = 0
    )
    {
        if ( isPolymorphicReducedAA( polymorphicState ))
        {
            auto stateMutations = generateReducedAAMutations<AAGrouping>( polymorphicState );
            return std::accumulate( stateMutations.cbegin(),
                                    stateMutations.cend(), acc,
                                    [fn](
                                            double acc,
                                            char stateMutation
                                    ) {
                                        return acc + fn( stateMutation );
                                    } );
        } else
        {
            return acc + fn( polymorphicState );
        }
    }

    template<typename AAGrouping>
    static double polymorphicSummer(
            std::string_view polymorphicContext,
            char polymorphicState,
            std::function<double(
                    std::string_view,
                    char
            )> fn,
            double acc = 0
    )
    {
        if ( isPolymorphicReducedSequence<AAGrouping::StatesN>( polymorphicContext ))
        {
            auto contextMutations =
                    generateReducedPolymorphicSequenceOutcome<AAGrouping>( polymorphicContext );
            return std::accumulate( contextMutations.cbegin(), contextMutations.cend(), acc,
                                    [=](
                                            double acc,
                                            std::string_view contextMutation
                                    ) {
                                        return acc + polymorphicSummer<AAGrouping>(
                                                polymorphicState,
                                                [contextMutation, fn]( char state ) {
                                                    return fn( contextMutation,
                                                               state );
                                                } );
                                    } );
        } else
        {
            return acc + polymorphicSummer<AAGrouping>( polymorphicState, [=]( char state ) {
                return fn( polymorphicContext, state );
            } );
        }
    }

    template<typename AAGrouping, typename AppliedFn>
    static void polymorphicApply(
            char polymorphicState,
            AppliedFn fn
    )
    {
        if ( isPolymorphicReducedAA( polymorphicState ))
        {
            auto stateMutations = generateReducedAAMutations<AAGrouping>( polymorphicState );
            std::for_each( stateMutations.cbegin(), stateMutations.cend(), fn );
        } else
        {
            fn( polymorphicState );
        }
    }

    template<typename AAGrouping, typename AppliedFn>
    static void polymorphicApply(
            std::string_view polymorphicContext,
            char polymorphicState,
            AppliedFn fn
    )
    {
        static constexpr auto States = AAGrouping::StatesN;
        if ( isPolymorphicReducedSequence<States>( polymorphicContext ))
        {
            auto contextMutations = generateReducedPolymorphicSequenceOutcome<AAGrouping>(
                    polymorphicContext );
            std::for_each( contextMutations.cbegin(), contextMutations.cend(),
                           [=]( std::string_view contextMutation ) {
                               polymorphicApply<AAGrouping>( polymorphicState, [=]( char state ) {
                                   fn( contextMutation, state );
                               } );
                           } );
        } else
        {
            polymorphicApply<AAGrouping>( polymorphicState, [=]( char state ) {
                fn( polymorphicContext, state );
            } );
        }
    }
};


#endif //MARKOVIAN_FEATURES_SEQUENCEENTRY_HPP
