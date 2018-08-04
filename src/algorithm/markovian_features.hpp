#ifndef MARKOVIAN_FEATURES_HPP
#define MARKOVIAN_FEATURES_HPP

#include <set>
#include <list>
#include <type_traits>
#include <variant>

#include <fmt/format.h>

#include "common.hpp"
#include "UniRefEntry.hpp"
#include "Timers.hpp"
#include "ConfusionMatrix.hpp"
#include "similarities.hpp"
#include "aminoacids_grouping.hpp"

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

        template<typename Distance>
        inline auto distance( const KernelUnit &unit ) const
        {
            return Distance::measure( _buffer.cbegin(), _buffer.cend(),
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


template<typename Grouping, typename Criteria>
class ConfiguredPipeline
{
    using PriorityQueue = typename MatchSet<Criteria>::Queue;
    using MarkovianProfile = MarkovianKernel<Grouping>;
    using KernelUnit = typename MarkovianProfile::KernelUnit;
    using MarkovianProfiles = std::map<std::string, MarkovianProfile>;

    static constexpr const char *LOADING = "loading";
    static constexpr const char *PREPROCESSING = "preprocessing";
    static constexpr const char *TRAINING = "training";
    static constexpr const char *CLASSIFICATION = "classification";

public:
    static std::vector<UniRefEntry>
    reducedAlphabetEntries( const std::vector<UniRefEntry> &entries )
    {
        return UniRefEntry::reducedAlphabetEntries<Grouping>( entries );
    }

    static double totalDistance( const MarkovianProfile &query,
                                 const MarkovianProfile &target )
    {
        double sum = 0;
        for (const auto &[rowId, row] : query.kernel())
        {
            try
            {
                auto &unit1 = row;
                auto &unit2 = target.kernel().at( rowId );
                sum += unit1.template distance<Criteria>( unit2 );
            } catch (const std::out_of_range &e)
            {

            }
        }
        return sum;
    }

    static PriorityQueue findSimilarities( const MarkovianProfile &query,
                                           const MarkovianProfiles &targets,
                                           size_t kNearest = 5 )
    {
        PriorityQueue matchSet;
        for (const auto &[clusterId, clusterProfile] : targets)
        {
            double distance = totalDistance( query, clusterProfile );
            matchSet.insert( {clusterId, distance} );

            if ( matchSet.size() > kNearest )
                matchSet.erase( matchSet.begin());
        }
        return matchSet;
    }


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
            ClassificationCandidates<Criteria> result{"", findSimilarities( p, targets )};
            classifications.emplace_back( result );
        }
        return classifications;
    }

    static std::vector<ClassificationCandidates<Criteria>>
    classify_VALIDATION(
            const std::vector<UniRefEntry> &queries,
            const MarkovianProfiles &targets )
    {
        const int order = targets.begin()->second.order();
        std::vector<ClassificationCandidates<Criteria>> classifications;
        size_t truePositive = 0;
        size_t tested = 0;
        for (const auto &unirefItem : queries)
        {
            MarkovianProfile p( order );
            p.train( {unirefItem.getSequence()} );
            ClassificationCandidates<Criteria> result{unirefItem.getClusterName(),
                                                      findSimilarities( p, targets )};
            classifications.emplace_back( result );
            truePositive += result.trueClusterFound();
            ++tested;
            if ((tested * 100) / queries.size() - ((tested - 1) * 100) / queries.size() > 0 )
            {
                fmt::print( "[progress:{}%][accuracy:{}]\n",
                            float( tested * 100 ) / queries.size(),
                            float( truePositive ) / tested );
            }
        }
        return classifications;
    }


    void runPipeline_VALIDATION( std::vector<UniRefEntry> entries,
                                 int order,
                                 double testPercentage,
                                 double threshold )
    {
        auto[trainingClusters, test] = Timers::reported_invoke_s( [&]() {

            fmt::print( "[All Sequences:{}]\n", entries.size());

            auto[test, training] = (threshold > 0) ?
                                   UniRefEntry::separationExcludingClustersWithLowSequentialData( entries,
                                                                                                  testPercentage,
                                                                                                  threshold ) :
                                   subsetRandomSeparation( entries, testPercentage );

            test = reducedAlphabetEntries( test );
            training = reducedAlphabetEntries( training );

            fmt::print( "[Training Entries:{}][Test Entries:{}][Test Ratio:{}]\n",
                        training.size(), test.size(), float( test.size()) / entries.size());

            entries.clear();

            auto trainingClusters = UniRefEntry::groupSequencesByUniRefClusters( training );
            return std::make_pair( trainingClusters, test );
        }, PREPROCESSING );


        fmt::print( "[Training Clusters:{}]\n", trainingClusters.size());
        auto trainedProfiles = Timers::reported_invoke_s( [&]() {
            return markovianTraining( trainingClusters, order );
        }, TRAINING );

        auto classificationResults = Timers::reported_invoke_s( [&]() {
            return classify_VALIDATION( test, trainedProfiles );
        }, CLASSIFICATION );

        std::set<std::string> labels;
        for (const auto &[k, v] : trainedProfiles)
            labels.insert( k );
        for (const auto &t : test)
            labels.insert( t.getClusterName());

        ConfusionMatrix c( labels );
        std::unordered_map<long, size_t> histogram;
        for (const auto &classification : classificationResults)
        {
            ++histogram[classification.trueClusterRank()];
            c.countInstance( classification.bestMatch(), classification.trueCluster );
        }
        c.printReport();

        fmt::print( "True Classification Histogram:\n" );

        for (auto &[k, v] : histogram)
        {
            if ( k == -1 ) continue;
            fmt::print( "Rank:{:<10}Count:{}\n", k, v );
        }
    }
};


/**
 * credits: https://stackoverflow.com/questions/46831599/create-cartesian-product-expansion-of-two-variadic-non-type-template-parameter
 * @tparam ...
 */

template <typename, typename...> struct midProd {};

template <typename...>
struct ConfigurationCombination;

template <typename ... R>
struct ConfigurationCombination<std::variant<R...>>
{ using type = std::variant<R...>; };

template <typename ... R, typename i, typename ... cs, typename ... MpS>
struct ConfigurationCombination<std::variant<R...>, midProd<i, cs...>, MpS...>
{ using type = typename ConfigurationCombination<std::variant<R..., ConfiguredPipeline<i, cs>...>, MpS...>::type; };


template <typename, typename>
struct magic;

template <typename ... is, typename ... cs>
struct magic<AAGroupingList<is...>, CriteriaList<cs...>>
{ using type = typename ConfigurationCombination<std::variant<>, midProd<is, cs...>...>::type; };


using PipelineVariant = typename magic<SuppotedAAGrouping, SupportedCriteria>::type;


template< typename AAGrouping >
PipelineVariant getConfiguredPipeline( CriteriaEnum criteria )
{
    switch (criteria)
    {
        case CriteriaEnum::ChiSquared :
        {
            return ConfiguredPipeline< AAGrouping, ChiSquared>();
        }
            break;
    }
};


PipelineVariant getConfiguredPipeline( AminoAcidGroupingEnum grouping, CriteriaEnum criteria )
{
    switch (grouping)
    {
        case AminoAcidGroupingEnum::DIAMOND11 :
        {
            return getConfiguredPipeline< AAGrouping_DIAMOND11>( criteria );
        }
            break;
        case AminoAcidGroupingEnum::OLFER8 :
        {
            return getConfiguredPipeline< AAGrouping_OLFER8>( criteria );
        }
            break;
        case AminoAcidGroupingEnum::OLFER15 :
        {
            return getConfiguredPipeline< AAGrouping_OLFER15>( criteria );
        }
            break;
    }
}

PipelineVariant getConfiguredPipeline( const std::string &groupingName,
                                       const std::string &criteria )
{
    const AminoAcidGroupingEnum groupingLabel = GroupingLabels.at( groupingName );
    const CriteriaEnum criteriaLabel = CriteriaLabels.at( criteria );
    return getConfiguredPipeline( groupingLabel, criteriaLabel );
}



#endif
