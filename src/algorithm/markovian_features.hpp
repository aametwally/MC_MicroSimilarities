#ifndef MARKOVIAN_FEATURES_HPP
#define MARKOVIAN_FEATURES_HPP

#include <set>
#include <list>
#include <type_traits>
#include <variant>
#include <tuple>
#include <typeinfo>
#include <cxxabi.h>

#include <fmt/format.h>

#include "common.hpp"
#include "LabeledEntry.hpp"
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
class MarkovianKernels
{
public:
    using Order = int8_t;
    using KernelID = size_t;

    static constexpr size_t StatesN = AAGrouping::StatesN;
    static constexpr std::array<char, StatesN> ReducedAlphabet = reducedAlphabet<StatesN>();
    static constexpr std::array<char, 256> ReducedAlphabetIds = reducedAlphabetIds( AAGrouping::Grouping );
    static constexpr Order MinOrder = 2;
    static constexpr double PseudoCounts = double( 1 ) / StatesN;

    using Buffer = std::array<double, StatesN>;
    using BufferIterator = typename Buffer::iterator;
    using BufferConstIterator = typename Buffer::const_iterator;

public:
    class Kernel
    {
    public:
        explicit Kernel( double pseudoCount = PseudoCounts )
                : _hits( 0 )
        {
            _buffer.fill( pseudoCount );
        }

        inline BufferConstIterator begin() const
        {
            return _buffer.begin();
        }

        inline BufferConstIterator end() const
        {
            return _buffer.end();
        }

        inline BufferConstIterator cbegin() const
        {
            return _buffer.cbegin();
        }

        inline BufferConstIterator cend() const
        {
            return _buffer.cend();
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

        inline double sum() const
        {
            return std::accumulate( _buffer.cbegin(), _buffer.cend(),
                                    double( 0 ));
        }

        inline void normalize()
        {
            const auto total = sum();
            for (auto &p : _buffer)
                p /= total;
        }

        constexpr static auto maxInformation()
        {
            return maxInformation_UNIFORM( StatesN );
        }

        inline double information() const
        {
            return entropy( _buffer.cbegin(), _buffer.cend());
        }

        inline size_t hits() const
        {
            return _hits;
        }

        std::string toString() const
        {
            return io::join2string( _buffer, " " );
        }

        Kernel &operator+=( const Kernel &other )
        {
            for (auto i = 0; i < StatesN; ++i)
                _buffer[i] += (other._buffer[i]);
            return *this;
        }

        Kernel operator*( double factor ) const
        {
            Kernel newKernel;
            for (auto i = 0; i < StatesN; ++i)
                newKernel._buffer[i] = _buffer[i] * factor;
            return newKernel;
        }

        static inline double dot( const Kernel &k1, const Kernel &k2 )
        {
            double sum = 0;
            for (auto i = 0; i < StatesN; ++i)
                sum += k1._buffer[i] * k2._buffer[i];
            return sum;
        }

        static inline double magnitude( const Kernel &k )
        {
            double sum = 0;
            for (auto i = 0; i < StatesN; ++i)
                sum += k._buffer[i] * k._buffer[i];
            return std::sqrt( sum );
        }

        inline double magnitude() const
        {
            return magnitude( *this );
        }

        double operator*( const Kernel &other ) const
        {
            return dot( *this, other );
        }

    protected:
        Buffer _buffer;
        size_t _hits;
    };


    template<typename T>
    using IsoKernelsAssociativeCollection = std::unordered_map<KernelID, T>;

    template<typename T>
    using HeteroKernelsAssociativeCollection = std::unordered_map<Order, IsoKernelsAssociativeCollection<T >>;

    using IsoKernels = IsoKernelsAssociativeCollection<Kernel>;
    using HeteroKernels = HeteroKernelsAssociativeCollection<Kernel>;
    using HeteroKernelsFeatures = HeteroKernelsAssociativeCollection<double>;
public:
    explicit MarkovianKernels( Order order ) :
            _maxOrder( order ),
            _hits( 0 )
    {
        assert( order >= MinOrder );
    }

    static IsoKernels
    filterPercentile( const std::unordered_map<size_t, Kernel> &filteredKernel,
                      float percentile )
    {
        std::vector<std::pair<KernelID, Kernel >> v;
        for (const auto &p : filteredKernel) v.push_back( p );

        auto cmp = []( const std::pair<size_t, Kernel> &p1, const std::pair<size_t, Kernel> &p2 ) {
            return p1.second.hits() > p2.second.hits();
        };

        size_t percentileTailIdx = filteredKernel.size() * percentile;
        std::nth_element( v.begin(), v.begin() + percentileTailIdx,
                          v.end(), cmp );

        IsoKernels filteredKernel2( v.begin(), v.begin() + percentileTailIdx );

        return filteredKernel2;
    }

    void train( const std::vector<std::string> &sequences )
    {
        for (const auto &s : sequences)
            _countInstance( s );

        for (Order order = MinOrder; order <= _maxOrder; ++order)
            for (auto &[id, kernel] : _kernels.at( order ))
                kernel.normalize();
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
        for (const auto &[id, kernel] : kernels())
            kernelFile << kernel.toString() << std::endl;
        kernelFile.close();
    }

    const IsoKernels &kernels( Order order ) const
    {
        return _kernels.at( order );
    }

    Order maxOrder() const
    {
        return _maxOrder;
    }

private:
    void _incrementInstance( std::string::const_iterator from,
                             std::string::const_iterator until,
                             Order order )
    {
        assert( from != until );
        KernelID id = _sequence2ID( from, until );
        auto c = ReducedAlphabetIds.at( size_t( *(until)));
        _kernels[order][id].increment( c );
    }

    void _countInstance( const std::string &sequence )
    {
        ++_hits;
        for (auto order = MinOrder; order <= _maxOrder; ++order)
            for (auto i = 0; i < sequence.size() - order; ++i)
                _incrementInstance( sequence.cbegin() + i, sequence.cbegin() + i + order, order );
    }

    static constexpr inline KernelID _char2ID( char a )
    {
        assert( a >= ReducedAlphabet.front());
        return KernelID( a - ReducedAlphabet.front());
    }

    static constexpr inline char _id2Char( KernelID id )
    {
        assert( id <= 128 );
        return char( id + ReducedAlphabet.front());
    }

    static KernelID _sequence2ID( const std::string &s )
    {
        return _sequence2ID( s.cbegin(), s.cend());
    }

    static KernelID _sequence2ID( std::string::const_iterator from,
                                  std::string::const_iterator until,
                                  KernelID init = 0 )
    {
        KernelID code = init;
        for (auto it = from; it != until; ++it)
            code = code * StatesN + _char2ID( *it );
        return code;
    }

    static std::string _id2Sequence( KernelID id, const size_t size , std::string &&acc = "" )
    {
        if ( acc.size() == size ) return acc;
        else return _id2Sequence( id / StatesN, size , _id2Char( id % StatesN ) + acc );
    }

private:
    const Order _maxOrder;
    HeteroKernels _kernels;
    size_t _hits;
};

struct Voting
{
};
struct Accumulative
{
};
enum class StrategyEnum
{
    Voting,
    Accumulative
};
static const std::map<std::string, StrategyEnum> ClassificationStrategyLabel = {
        {"voting", StrategyEnum::Voting},
        {"acc",    StrategyEnum::Accumulative}
};

template<typename Grouping, typename Criteria, typename Strategy>
class ConfiguredPipeline
{
public:

private:
    using PriorityQueue = typename MatchSet<Criteria>::Queue;
    using VotingPQ = typename MatchSet<Score>::Queue;
    using MarkovianProfile = MarkovianKernels<Grouping>;
    using Kernel = typename MarkovianProfile::Kernel;
    using MarkovianProfiles = std::map<std::string, MarkovianProfile>;
    using KernelID = typename MarkovianProfile::KernelID;
    using Order = typename MarkovianProfile::Order;

    using HeteroKernels =  typename MarkovianProfile::HeteroKernels;
    using HeteroKernelsFeatures =  typename MarkovianProfile::HeteroKernelsFeatures;


    static constexpr Order MinOrder = MarkovianProfile::MinOrder;

    static constexpr const char *LOADING = "loading";
    static constexpr const char *PREPROCESSING = "preprocessing";
    static constexpr const char *TRAINING = "training";
    static constexpr const char *CLASSIFICATION = "classification";

    using LeaderBoard = typename std::conditional<std::is_same<Voting, Strategy>::value, ClassificationCandidates<Score>, ClassificationCandidates<Criteria> >::type;

public:


    static std::vector<LabeledEntry>
    reducedAlphabetEntries( const std::vector<LabeledEntry> &entries )
    {
        return LabeledEntry::reducedAlphabetEntries<Grouping>( entries );
    }

    static VotingPQ findSimilarityByVoting(
            const MarkovianProfile &query,
            const MarkovianProfiles &targets,
            const HeteroKernelsFeatures &histogramsRadius,
            const HeteroKernels &meanHistogram )
    {
        Order mxOrder = maxOrder( targets );
        std::map<std::string, double> accumulator;
        for (const auto &[id, kernel] : query.kernels( mxOrder ))
        {
            using PQ = typename MatchSet<Criteria>::Queue;
            PQ pq( targets.size());
            double p1 = double( kernel.hits()) / query.hits();
            auto factor = [&, id]() {
                return p1 * p1 * histogramsRadius.at( mxOrder ).at( id );
            };

            auto &kernel1 = kernel;
            for (const auto &[clusterName, profile] : targets)
            {
                try
                {
                    auto &kernel2 = profile.kernels( mxOrder ).at( id );
                    double value = Criteria::measure( kernel1, kernel2 );
                    pq.emplace( clusterName, value );
                } catch (const std::out_of_range &e)
                {


                }
            }
            if ( !pq.empty())
            {
                accumulator[pq.top().getLabel()] += factor();
            }
        }
        std::pair<std::string, double> maxVotes = *accumulator.begin();
        VotingPQ vPQ( targets.size());
        for (const auto &[id, votes]: accumulator)
            vPQ.emplace( id, votes );
        return vPQ;
    }

    static PriorityQueue findSimilarities( const MarkovianProfile &query,
                                           const MarkovianProfiles &targets,
                                           const HeteroKernelsFeatures &histogramsRadius,
                                           const HeteroKernels &meanHistogram )
    {
        PriorityQueue matchSet( targets.size());
        Order mxOrder = maxOrder( targets );
        for (const auto &[clusterId, profile] : targets)
        {
            double sum = 0;
            for (Order order = MinOrder; order <= mxOrder; ++order)
                for (const auto &[id, kernel] : query.kernels( order ))
                {
                    try
                    {
                        auto &unit1 = kernel;
                        auto &unit2 = profile.kernels( order ).at( id );
                        sum += Criteria::measure( unit1, unit2 );
                    } catch (const std::out_of_range &e)
                    {}
                }
            matchSet.emplace( clusterId, sum );
        }
        return matchSet;
    }

    template<typename S = Strategy>
    static typename std::enable_if<std::is_same<S, Voting>::value, VotingPQ>::type
    findSimilarity( const MarkovianProfile &query,
                    const MarkovianProfiles &targets,
                    const HeteroKernelsFeatures &histogramsRadius,
                    const HeteroKernels &meanHistogram )
    {
        return findSimilarityByVoting( query, targets, histogramsRadius, meanHistogram );
    }

    template<typename S = Strategy>
    static typename std::enable_if<std::is_same<S, Accumulative>::value, PriorityQueue>::type
    findSimilarity( const MarkovianProfile &query,
                    const MarkovianProfiles &targets,
                    const HeteroKernelsFeatures &histogramsRadius,
                    const HeteroKernels &meanHistogram )
    {
        return findSimilarities( query, targets, histogramsRadius, meanHistogram );
    }

    static MarkovianProfiles
    markovianTraining( const std::map<std::string, std::vector<std::string >> &training,
                       Order markovianOrder )
    {
        MarkovianProfiles trainedProfiles;

        for (const auto &[id, sequences] : training)
        {
            MarkovianKernels<Grouping> kernel( markovianOrder );
            kernel.train( sequences );
            trainedProfiles.emplace( id, std::move( kernel ));
        }
        return trainedProfiles;
    }

    static Order maxOrder( const MarkovianProfiles &profiles )
    {
        return profiles.cbegin()->second.maxOrder();
    }

    static std::map<std::string, double>
    clustersWeight( const MarkovianProfiles &profiles )
    {
        std::map<std::string, double> weights;
        size_t sum = 0;

        for (const auto &[_, profile] : profiles)
            sum += profile.hits();

        for (auto &[c, profile] : profiles)
            weights[c] = double( profile.hits()) / sum;

        return weights;
    }

    static HeteroKernels
    meanHistograms( const MarkovianProfiles &profiles,
                    const std::map<std::string, HeteroKernelsFeatures> &kernelWeights )
    {
        const Order mxOrder = maxOrder( profiles );
        HeteroKernels means;

        for (const auto &[cluster, profile] : profiles)
        {
            const auto &weights = kernelWeights.at( cluster );
            for (auto order = MinOrder; order <= mxOrder; ++order)
                for (const auto &[id, kernel] : profile.kernels( order ))
                    means[order][id] += (kernel * weights.at( order ).at( id ));
        }
        return means;
    }

    static HeteroKernels
    meanHistograms( const MarkovianProfiles &profiles )
    {
        return meanHistograms( profiles, histogramWeights( profiles ));
    }

    static std::map<std::string, HeteroKernelsFeatures>
    histogramWeights( const MarkovianProfiles &profiles )
    {
        const Order mxOrder = maxOrder( profiles );

        std::map<std::string, HeteroKernelsFeatures> histogramWeights;
        std::unordered_map<Order, std::set<KernelID >> scannedIDs;

        for (const auto &[cluster, profile] : profiles)
        {
            auto &weights = histogramWeights[cluster];
            for (auto order = MinOrder; order <= mxOrder; ++order)
            {
                for (const auto &[id, histogram] : profile.kernels( order ))
                {
                    weights[order][id] = histogram.hits();
                    scannedIDs[order].insert( id );
                }
            }
        }

        for (auto order = MinOrder; order <= mxOrder; ++order)
            for (auto id : scannedIDs.at( order ))
            {
                double sum = 0;
                for (auto &[cluster, weights] : histogramWeights)
                    sum += weights[order][id];

                for (auto &[cluster, weights] : histogramWeights)
                    weights.at( order ).at( id ) /= sum;
            }

        return histogramWeights;
    }

    static std::unordered_map<KernelID, double>
    histogramExclusiveness( const MarkovianProfiles &profiles, const std::map<std::string, double> &clustersWeights )
    {
        const Order mxOrder = maxOrder( profiles );

        const size_t k = profiles.size();
        std::unordered_map<KernelID, std::vector<double> > kernelWeights;
        std::unordered_map<KernelID, double> significance;

        for (const auto &[cluster, profile] : profiles)
            for (const auto &[id, kernel] : profile.kernels())
                kernelWeights[id].push_back( kernel.hits() / clustersWeights.at( cluster ));

        for (auto &[id, kernelW] : kernelWeights)
        {
            double total = std::accumulate( kernelW.cbegin(), kernelW.cend(), double( 0 ));
            for (auto &p: kernelW) p /= total;
            significance[id] = informationGain_UNIFORM( kernelW.cbegin(), kernelW.cend(), k );
        }
        return significance;
    }

    static HeteroKernelsFeatures
    informationRadius( const MarkovianProfiles &profiles )
    {
        const Order mxOrder = maxOrder( profiles );

        const size_t k = profiles.size();
        std::map<std::string, HeteroKernelsFeatures> histogramsWeights = histogramWeights( profiles );

        HeteroKernelsFeatures meanEntropies;
        HeteroKernels meanHistogram;
        HeteroKernelsFeatures radius;

        for (const auto &[cluster, profile] : profiles)
        {
            const auto &weights = histogramsWeights.at( cluster );
            for (auto order = MinOrder; order <= mxOrder; ++order)
                for (const auto &[id, histogram] : profile.kernels( order ))
                {
                    double b = weights.at( order ).at( id );
                    meanEntropies[order][id] += (histogram.information() * b);
                    meanHistogram[order][id] += (histogram * b);
                }
        }

        for (auto order = MinOrder; order <= mxOrder; ++order)
            for (auto &[id, meanEntropy] : meanEntropies.at( order ))
            {
                radius[order][id] = meanHistogram.at( order ).at( id ).information() - meanEntropy;
            }
        return radius;
    }


    static std::vector<LeaderBoard>
    classify( const std::vector<FastaEntry> &queries,
              const MarkovianProfiles &targets )
    {
        const int order = targets.begin()->second.order();
        std::vector<LeaderBoard> classifications;
        for (const auto &q : queries)
        {
            MarkovianProfile p( order );
            p.train( {q.getSequence()} );
            classifications.emplace_back( "", findSimilarity( p, targets ));
        }
        return classifications;
    }

    static std::vector<LeaderBoard>
    classify_VALIDATION(
            const std::vector<std::string> &queries,
            const std::vector<std::string> &trueLabels,
            const MarkovianProfiles &targets )
    {
        assert( queries.size() == trueLabels.size());
        const int order = targets.begin()->second.maxOrder();
        std::vector<LeaderBoard> classifications;
        size_t truePositive = 0;
        size_t tested = 0;
//        auto clustersWeights = clustersWeight( targets );
//        auto histogramsExclusiveness = histogramExclusiveness( targets, clustersWeights );
        auto histogramsRadius = informationRadius( targets );
        auto meanHistogram = meanHistograms( targets );
        for (auto i = 0; i < queries.size(); ++i)
        {
            MarkovianProfile p( order );
            p.train( {queries.at( i )} );
            classifications.emplace_back( trueLabels.at( i ),
                                          findSimilarity( p, targets, histogramsRadius,
                                                          meanHistogram ));
        }
        return classifications;
    }


    void runPipeline_VALIDATION( std::vector<LabeledEntry> &&entries,
                                 Order order,
                                 size_t k )
    {
        std::set<std::string> labels;
        for (const auto &entry : entries)
            labels.insert( entry.getLabel());

        using Folds = std::vector<std::vector<std::pair<std::string, std::string >>>;

        const Folds folds = Timers::reported_invoke_s( [&]() {
            fmt::print( "[All Sequences:{}]\n", entries.size());
            entries = reducedAlphabetEntries( entries );
            auto groupedEntries = LabeledEntry::groupSequencesByLabels( entries );

            auto labels = keys( groupedEntries );
            for (auto &l : labels) l = fmt::format( "{}({})", l, groupedEntries.at( l ).size());

            fmt::print( "[Clusters:{}][{}]\n",
                        groupedEntries.size(),
                        io::join( labels, "|" ));

            return kFoldStratifiedSplit( std::move( groupedEntries ), k );
        }, PREPROCESSING );


        auto extractTest = []( const std::vector<std::pair<std::string, std::string >> &items ) {
            std::vector<std::string> sequences, labels;
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

template<template<typename...> class C, typename ... Ts>
constexpr std::tuple<C<Ts...>> tupleExpand( std::tuple<Ts...> const & );

template<template<typename...> class C, typename ... Ts,
        template<typename...> class C0, typename ... Ls,
        typename ... Cs>
constexpr auto tupleExpand( std::tuple<Ts...> const &, C0<Ls...> const &,
                            Cs const &... cs )
-> decltype( std::tuple_cat(
        tupleExpand<C>( std::declval<std::tuple<Ts..., Ls>>(), cs... )... ));

template<typename ... Ts>
constexpr std::variant<Ts...> tupleToVariant( std::tuple<Ts...> const & );

template<template<typename...> class C, typename ... Ts>
struct MakeVariant
{
    using type = decltype( tupleToVariant( std::declval<
            decltype( tupleExpand<C>( std::declval<std::tuple<>>(),
                                      std::declval<Ts>()... ))>()));
};

template<template<typename...> class C, typename ... Ts>
using MakeVariantType = typename MakeVariant<C, Ts...>::type;




template<typename...>
struct StrategiesList
{
};
using SupportedStrategies = StrategiesList<Voting, Accumulative>;


using PipelineVariant = MakeVariantType<ConfiguredPipeline,
        SupportedAAGrouping,
        SupportedCriteria,
        SupportedStrategies>;

template<typename AAGrouping, typename Criteria>
PipelineVariant getConfiguredPipeline( StrategyEnum strategy )
{
    switch (strategy)
    {
        case StrategyEnum::Accumulative :
            return ConfiguredPipeline<AAGrouping, Criteria, Accumulative>();
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
        case AminoAcidGroupingEnum::NoGrouping20:
            return getConfiguredPipeline<AAGrouping_NOGROUPING20>( criteria, strategy );
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
