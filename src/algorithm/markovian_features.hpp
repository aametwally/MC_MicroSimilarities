#ifndef MARKOVIAN_FEATURES_HPP
#define MARKOVIAN_FEATURES_HPP

#include <set>
#include <list>
#include <type_traits>
#include <variant>
#include <typeinfo>
#include <cxxabi.h>

#include <fmt/format.h>

#include "common.hpp"
#include "UniRefEntry.hpp"
#include "Timers.hpp"
#include "ConfusionMatrix.hpp"
#include "similarities.hpp"
#include "aminoacids_grouping.hpp"
#include "CrossValidationStatistics.hpp"
#include "crossvalidation.hpp"

auto geometricDistribution( double p )
{
    return [p]( double exponent ) {
        return pow( p, exponent );
    };
}

template<typename T>
auto inverseFunction()
{
    return []( T n ) {
        return T( 1 ) / n;
    };
}

template<typename AAGrouping>
class MarkovianKernel
{
public:
    static constexpr size_t StatesN = AAGrouping::StatesN;
    static constexpr std::array<char, StatesN> ReducedAlphabet = reducedAlphabet<StatesN>();
    static constexpr std::array<char, 256> ReducedAlphabetIds = reducedAlphabetIds( AAGrouping::Grouping );

public:
    class KernelUnit
    {
    public:
        explicit KernelUnit( double pseudoCount = double{0.01} )
                : _hits( 0 )
        {
            _buffer.fill( pseudoCount );
        }


        inline auto &at( char state ) const
        {
            return _buffer.at( state );
        }

        inline void increment( char state )
        {
            ++_hits;
            ++_buffer.at( state );
        }

        inline void increment( char state, double val )
        {
            ++_hits;
            _buffer.at( state ) += val;
        }

        inline auto sum() const
        {
            return std::accumulate( _buffer.cbegin(), _buffer.cend(),
                                    double{0} );
        }

        inline void normalize()
        {
            if ( !isPristine())
            {
                const auto total = sum();
                for (auto &p : _buffer)
                    p /= total;
            } else
            {
                constexpr double p = double{1} / StatesN;
                std::fill( _buffer.begin(), _buffer.end(), p );
            }
        }

        constexpr static auto maxInformation()
        {
            double ent{0};
            constexpr double p = double{1} / StatesN;
            for (auto i = 0; i < StatesN; ++i)
                ent += p * log2( p );
            return ent;
        }

        inline auto information() const
        {
            double ent{0};
            for (auto p : _buffer)
                ent += p * log2( p );
            return ent;
        }

        template<typename Criteria>
        inline auto similarity( const KernelUnit &unit ) const
        {
            return Criteria::measure( _buffer.cbegin(), _buffer.cend(),
                                      unit._buffer.cbegin(), unit._buffer.cend());
        }


        inline size_t hits() const
        {
            return _hits;
        }

        inline bool isPristine() const
        {
            return _hits == 0;
        }

        std::string toString() const
        {
            return io::join2string( _buffer, " " );
        }

    protected:
        std::array<double, StatesN> _buffer;
        size_t _hits;
    };

public:
    explicit MarkovianKernel( int order ) :
            _order( order ),
            _hits( 0 )
    {
        assert( order > 0 );
    }

    static std::unordered_map<size_t, KernelUnit>
    filterPercentile( const std::unordered_map<size_t, KernelUnit> &filteredKernel,
                      float percentile )
    {
        std::vector<std::pair<size_t, KernelUnit >> v;
        for (const auto &p : filteredKernel) v.push_back( p );

        auto cmp = []( const std::pair<size_t, KernelUnit> &p1, const std::pair<size_t, KernelUnit> &p2 ) {
            return p1.second.hits() > p2.second.hits();
        };

        size_t percentileTailIdx = filteredKernel.size() * percentile;
        std::nth_element( v.begin(), v.begin() + percentileTailIdx,
                          v.end(), cmp );

        std::unordered_map<size_t, KernelUnit> filteredKernel2( v.begin(), v.begin() + percentileTailIdx );

        return filteredKernel2;
    }

    void train( const std::vector<std::string> &sequences )
    {
        for (const auto &s : sequences)
            _countInstance( s );

        for (auto &[rowId, row] : _kernel)
            row.normalize();
    }

    size_t hits() const
    {
        return _hits;
    }

    void toFiles( const std::string &dir,
                  const std::string &prefix,
                  const std::string &id ) const
    {
        std::ofstream kernelFile;
        std::vector<std::string> names1 = {prefix, "profile", id};
        kernelFile.open( dir + "/" + io::join( names1, "_" ) + ".array" );
        for (const auto &u : _kernel)
            kernelFile << u.toString() << std::endl;
        kernelFile.close();
    }

    const std::unordered_map<size_t, KernelUnit> &kernel() const
    {
        return _kernel;
    }

    int order() const
    {
        return _order;
    }

private:
    void _incrementInstance( std::string::const_iterator from,
                             std::string::const_iterator until )
    {
        assert( from != until );
        auto index = _sequence2Index( from, until - 1 );
        auto c = ReducedAlphabetIds.at( *(until - 1));
        _kernel[index].increment( c );
    }

    void _countInstance( const std::string &sequence )
    {
        ++_hits;
        for (auto i = 0; i < sequence.size() - (_order + 1); ++i)
            _incrementInstance( sequence.cbegin() + i, sequence.cbegin() + i + _order + 1 );
    }

    static size_t _sequence2Index( std::string::const_iterator from,
                                   std::string::const_iterator until,
                                   size_t init = 0 )
    {
        size_t code = init;
        for (auto it = from; it != until; ++it)
            code = code * StatesN + *it - ReducedAlphabet.front();
        return code;
    }


private:
    const int _order;
    std::unordered_map<size_t, KernelUnit> _kernel;
    size_t _hits;
};

struct Voting
{
};
struct TotalDistance
{
};
enum class StrategyEnum
{
    Voting,
    TotalDistance
};
static const std::map<std::string, StrategyEnum> ClassificationStrategyLabel = {
        {"voting",    StrategyEnum::Voting},
        {"totaldist", StrategyEnum::TotalDistance}
};

template<typename Grouping, typename Criteria, typename Strategy = Voting>
class ConfiguredPipeline
{
public:

private:
    using PriorityQueue = typename MatchSet<Criteria>::Queue;
    using VotingPQ = typename MatchSet<Score>::Queue;
    using MarkovianProfile = MarkovianKernel<Grouping>;
    using KernelUnit = typename MarkovianProfile::KernelUnit;
    using MarkovianProfiles = std::map<std::string, MarkovianProfile>;

    static constexpr const char *LOADING = "loading";
    static constexpr const char *PREPROCESSING = "preprocessing";
    static constexpr const char *TRAINING = "training";
    static constexpr const char *CLASSIFICATION = "classification";

    using Prediction = LeaderBoard;
private:


public:
    static std::vector<UniRefEntry>
    reducedAlphabetEntries( const std::vector<UniRefEntry> &entries )
    {
        return UniRefEntry::reducedAlphabetEntries<Grouping>( entries );
    }

    static double totalSimilarityMeasure( const MarkovianProfile &query,
                                          const MarkovianProfile &target )
    {
        double sum = 0;
        for (const auto &[rowId, row] : query.kernel())
        {
            try
            {
                auto &unit1 = row;
                auto &unit2 = target.kernel().at( rowId );
                sum += unit1.template similarity<Criteria>( unit2 );
            } catch (const std::out_of_range &e)
            {}
        }
        return sum;
    }


    static VotingPQ findSimilarityByVoting(
            const MarkovianProfile &query,
            const MarkovianProfiles &targets,
            size_t kNearest = 5 )
    {
        VotingPQ vPQ( kNearest );

        std::map<std::string, double> counter;
        for (const auto &[rowId, row] : query.kernel())
        {
            using PQ = typename MatchSet<Criteria>::Queue;
            PQ pq( 3 );
            double p1 = double( row.hits()) / query.hits();

            auto &unit1 = row;
            for (const auto &[clusterName, profile] : targets)
            {
                try
                {
                    auto &unit2 = profile.kernel().at( rowId );
                    double value = unit1.template similarity<Criteria>( unit2 );
                    pq.insert( {clusterName, value} );
                } catch (const std::out_of_range &e)
                {

                }
            }
            if ( !pq.empty())
            {
                const MarkovianProfile &topTarget = targets.at( pq.top().id );
                auto &targetRow = topTarget.kernel().at( rowId );
                double p2 = double( targetRow.hits()) / topTarget.hits();
                counter[pq.top().id] += 1;
//                counter[pq.top().id] += 1;
            }
        }
        std::pair<std::string, double> maxVotes = *counter.begin();
        for (const auto &[id, votes]: counter)
            vPQ.insert( {id, votes} );
        return vPQ;
    }

    static PriorityQueue findSimilarities( const MarkovianProfile &query,
                                           const MarkovianProfiles &targets,
                                           size_t kNearest = 5 )
    {
        PriorityQueue matchSet( kNearest );
        for (const auto &[clusterId, clusterProfile] : targets)
        {
            double measure = totalSimilarityMeasure( query, clusterProfile );
            matchSet.insert( {clusterId, measure} );
        }
        return matchSet;
    }


    template<typename S, typename Enable = void>
    struct ClassificationStrategy;

    template<typename S>
    struct ClassificationStrategy<S, typename std::enable_if<std::is_same<Voting, S>::value>::type>
    {
        static constexpr auto findSimilarity = []( const MarkovianProfile &query,
                                                   const MarkovianProfiles &targets,
                                                   size_t kNearest = 5 ) {
            return findSimilarityByVoting( query, targets, kNearest );
        };
    };

    template<typename S>
    struct ClassificationStrategy<S, typename std::enable_if<std::is_same<TotalDistance, S>::value>::type>
    {
        static constexpr auto findSimilarity = []( const MarkovianProfile &query,
                                                   const MarkovianProfiles &targets,
                                                   size_t kNearest = 5 ) {
            return findSimilarities( query, targets, kNearest );
        };
    };

    using Classifier = ClassificationStrategy<Strategy>;

    static MarkovianProfiles
    markovianTraining( const std::map<std::string, std::vector<std::string >> &training,
                       int markovianOrder )
    {
        MarkovianProfiles trainedProfiles;

        for (const auto &[id, sequences] : training)
        {
            MarkovianKernel<Grouping> kernel( markovianOrder );
            kernel.train( sequences );
            trainedProfiles.emplace( id, std::move( kernel ));
        }
        return trainedProfiles;
    }

    static std::vector<ClassificationCandidates<Criteria>>
    classify( const std::vector<FastaEntry> &queries,
              const MarkovianProfiles &targets )
    {
        const int order = targets.begin()->second.order();
        std::vector<ClassificationCandidates<Criteria>> classifications;
        for (const auto &q : queries)
        {
            MarkovianProfile p( order );
            p.train( {q.getSequence()} );
            classifications.emplace_back( "", Classifier::findSimilarity( p, targets ));
        }
        return classifications;
    }

    static std::vector<ClassificationCandidates<Score>>
    classify_VALIDATION(
            const std::vector<std::string> &queries,
            const std::vector<std::string> &trueLabels,
            const MarkovianProfiles &targets )
    {
        assert( queries.size() == trueLabels.size());
        const int order = targets.begin()->second.order();
        std::vector<ClassificationCandidates<Score>> classifications;
        size_t truePositive = 0;
        size_t tested = 0;

        for (auto i = 0; i < queries.size(); ++i)
        {
            MarkovianProfile p( order );
            p.train( {queries.at( i )} );
            classifications.emplace_back( trueLabels.at( i ), Classifier::findSimilarity( p, targets ));
        }
        return classifications;
    }


    void runPipeline_VALIDATION( std::vector<UniRefEntry> &&entries,
                                 int order,
                                 size_t k )
    {
        std::set<std::string> labels;
        for (const auto &entry : entries)
            labels.insert( entry.getClusterName());

        using Folds = std::vector<std::vector<std::pair<std::string, std::string >>>;

        const Folds folds = Timers::reported_invoke_s( [&]() {
            fmt::print( "[All Sequences:{}]\n", entries.size());
            entries = reducedAlphabetEntries( entries );
            return kFoldStratifiedSplit( UniRefEntry::groupSequencesByUniRefClusters( entries ), k );
        }, PREPROCESSING );


        auto extractTest = []( const std::vector<std::pair<std::string, std::string >> &items ) {
            std::vector<std::string> sequences,labels;
            for (const auto item : items)
            {
                labels.push_back( item.first );
                sequences.push_back( item.second );
            }
            return std::make_pair( sequences , labels );
        };

        CrossValidationStatistics validation( k, labels );
        std::unordered_map<long, size_t> histogram;

        for (auto i = 0; i < k; ++i)
        {
            auto trainingClusters = joinFoldsExceptK( folds, i );
            auto [test,tLabels] = extractTest( folds.at( i ));
            auto trainedProfiles = markovianTraining( trainingClusters, order );
            auto classificationResults = classify_VALIDATION( test, tLabels, trainedProfiles );

            for (const auto &classification : classificationResults)
            {
                ++histogram[classification.trueClusterRank()];
                validation.countInstance( i, classification.bestMatch(), classification.trueCluster());
            }
        }

        validation.printReport();

        fmt::print( "True Classification Histogram:\n" );

        for (auto &[k, v] : histogram)
        {
            if ( k == -1 )
                fmt::print( "{:<20}Count:{}\n", "Unclassified", v );
            else
                fmt::print( "{:<20}Count:{}\n", fmt::format( "Rank:{}", k ), v );
        }
    }
};


/**
 * credits: https://stackoverflow.com/questions/46831599/create-cartesian-product-expansion-of-two-variadic-non-type-template-parameter
 * @tparam ...
 */


template<typename, typename...>
struct midProd
{
};

template<typename...>
struct magicH;

template<typename ... R>
struct magicH<std::variant<R...>>
{
    using type = std::variant<R...>;
};

template<typename ... R, typename G, typename ... Cs, typename ... MpS>
struct magicH<std::variant<R...>, midProd<G, Cs...>, MpS...>
{
    using type = typename magicH<std::variant<R..., ConfiguredPipeline<G, Cs>...>, MpS...>::type;
};


template<typename, typename>
struct magic;

template<typename ... Gs, typename ... Cs>
struct magic<AAGroupingList<Gs...>, CriteriaList<Cs...>>
{
    using type = typename magicH<std::variant<>, midProd<Gs, Cs...>...>::type;
};


template<typename...>
struct StrategiesList
{
};
using SupportedStrategies = StrategiesList<Voting, TotalDistance>;


using PipelineVariant = magic<SupportedAAGrouping, SupportedCriteria>::type;

template<typename AAGrouping, typename Criteria>
PipelineVariant getConfiguredPipeline( StrategyEnum strategy )
{
    switch (strategy)
    {
        case StrategyEnum::TotalDistance :
            return ConfiguredPipeline<AAGrouping, Criteria, Voting>();
        case StrategyEnum::Voting :
            return ConfiguredPipeline<AAGrouping, Criteria, Voting>();
    }
};

template<typename AAGrouping>
PipelineVariant getConfiguredPipeline( CriteriaEnum criteria, StrategyEnum strategy )
{
    switch (criteria)
    {
        case CriteriaEnum::ChiSquared :
            return getConfiguredPipeline<AAGrouping, ChiSquared>( strategy );
        case CriteriaEnum::Cosine :
            return getConfiguredPipeline<AAGrouping, Cosine>( strategy );
        case CriteriaEnum::KullbackLeiblerDiv:
            return getConfiguredPipeline<AAGrouping, KullbackLeiblerDivergence>( strategy );
    }
};


PipelineVariant getConfiguredPipeline( AminoAcidGroupingEnum grouping, CriteriaEnum criteria,
                                       StrategyEnum strategy )
{
    switch (grouping)
    {
        case AminoAcidGroupingEnum::DIAMOND11 :
            return getConfiguredPipeline<AAGrouping_DIAMOND11>( criteria, strategy );
        case AminoAcidGroupingEnum::OLFER8 :
            return getConfiguredPipeline<AAGrouping_OLFER8>( criteria, strategy );
        case AminoAcidGroupingEnum::OLFER15 :
            return getConfiguredPipeline<AAGrouping_OLFER15>( criteria, strategy );

    }
}

PipelineVariant getConfiguredPipeline( const std::string &groupingName,
                                       const std::string &criteria,
                                       const std::string &strategy )
{
    const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
    const CriteriaEnum criteriaLabel = CriteriaLabels.at( criteria );
    const StrategyEnum strategyLabel = ClassificationStrategyLabel.at( strategy );
    return getConfiguredPipeline( groupingLabel, criteriaLabel, strategyLabel );
}


#endif
