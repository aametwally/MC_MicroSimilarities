//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
#define MARKOVIAN_FEATURES_MARKOVCHAINS_HPP

#include "common.hpp"
#include "aminoacids_grouping.hpp"
#include "Histogram.hpp"


namespace MC {
    using Order = int8_t;
    using HistogramID = size_t;

    struct KernelIdentifier
    {
        explicit KernelIdentifier( Order o, HistogramID i ) : order( o ), id( i )
        {}

        Order order;
        HistogramID id;
    };

    using Selection = std::unordered_map<Order, std::set<HistogramID >>;
    using SelectionFlat = std::unordered_map<Order, std::vector<HistogramID >>;
    using SelectionOrdered = std::map<Order, std::set<HistogramID>>;

    constexpr double eps = std::numeric_limits<double>::epsilon();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr double inf = std::numeric_limits<double>::infinity();

    template<typename AAGrouping>
    class AbstractMC
    {
    public:
        static constexpr size_t StatesN = AAGrouping::StatesN;
        static constexpr std::array<char, StatesN> ReducedAlphabet = reducedAlphabet<StatesN>();
        static constexpr std::array<char, 256> ReducedAlphabetIds = reducedAlphabetIds( AAGrouping::Grouping );

        using Histogram = buffers::Histogram<StatesN>;
        using Buffer =  typename Histogram::Buffer;
        using BufferIterator =  typename Buffer::iterator;
        using BufferConstIterator =  typename Buffer::const_iterator;

    public:
        using IsoHistograms = std::unordered_map<HistogramID, Histogram>;
        using HeteroHistograms =  std::unordered_map<Order, IsoHistograms>;
        using HeteroHistogramsFeatures = std::unordered_map<Order, std::unordered_map<HistogramID, double>>;

        using ModelTrainer = std::function<std::unique_ptr<AbstractMC>( const std::vector<std::string> &,
                                                                        std::optional<std::reference_wrapper<const Selection >> selection )>;
        using HistogramsTrainer = std::function<std::optional<HeteroHistograms>( const std::vector<std::string> &,
                                                                                 std::optional<std::reference_wrapper<const Selection >> selection )>;

        using BackboneProfiles = std::map<std::string, std::unique_ptr<AbstractMC >>;
    public:

        template<typename Histograms>
        AbstractMC( Histograms &&histograms )
                : _histograms( std::forward<Histograms>( histograms ))
        {

        }

        AbstractMC() = default;

        size_t histogramsCount() const
        {
            size_t sum = 0;
            for (auto &[order, isoHistograms] : _histograms)
                sum += isoHistograms.size();
            return sum;
        }

        bool contains( Order order ) const
        {
            auto isoHistogramsIt = _histograms.find( order );
            return isoHistogramsIt != _histograms.cend();
        }

        bool contains( Order order, HistogramID id ) const
        {
            if ( auto isoHistogramsIt = _histograms.find( order ); isoHistogramsIt != _histograms.cend())
            {
                auto histogramIt = isoHistogramsIt->second.find( id );
                return histogramIt != isoHistogramsIt->second.cend();
            } else return false;
        }

        Selection featureSpace() const noexcept
        {
            Selection features;
            for (auto &[order, isoHistograms] : _histograms)
            {
                for (auto &[id, histogram] : isoHistograms)
                {
                    features[order].insert( id );
                }
            }
            return features;
        }



        static inline bool isReducedSequences( const std::vector<std::string> &sequences )
        {
            return std::all_of( sequences.cbegin(), sequences.cend(), isReducedSequence );
        }

        static inline bool isReducedSequence( std::string_view sequence )
        {
            for (auto c : sequence)
                if ( auto it = std::find( ReducedAlphabet.cbegin(), ReducedAlphabet.cend(), c ); it ==
                                                                                                 ReducedAlphabet.cend())
                    return false;
            return true;
        }

        virtual void setMinOrder( Order mnOrder ) = 0;
        virtual void setMaxOrder( Order mxOrder ) = 0;
        virtual void setRangedOrders( std::pair< Order , Order > range ) = 0;

        virtual void train( const std::vector<std::string> &sequences ) = 0;

        virtual double probability( std::string_view, char ) const = 0;

        virtual double propensity( std::string_view ) const = 0;

        std::reference_wrapper<const HeteroHistograms> histograms() const
        {
            return _histograms;
        }

        HeteroHistograms convertToHistograms()
        {
            return std::move( _histograms );
        }

        void setHistograms( HeteroHistograms &&histograms )
        {
            _histograms = std::move( histograms );
        }

        HeteroHistograms &&histograms()
        {
            return std::move( _histograms );
        }

        std::optional<std::reference_wrapper<const IsoHistograms>> histograms( Order distance ) const
        {
            if ( auto histogramsIt = _histograms.find( distance ); histogramsIt != _histograms.cend())
                return std::cref( histogramsIt->second );
            return std::nullopt;
        }

        std::optional<std::reference_wrapper<const Histogram>> histogram( Order distance, HistogramID id ) const
        {
            if ( auto histogramsOpt = histograms( distance ); histogramsOpt )
                if ( auto histogramIt = histogramsOpt.value().get().find( id ); histogramIt !=
                                                                                histogramsOpt.value().get().cend())
                    return std::cref( histogramIt->second );
            return std::nullopt;
        }

        void clear()
        {
            _histograms.clear();
        }

        template<typename Histograms>
        void set( Histograms &&histograms )
        {
            _histograms = std::forward<Histograms>( histograms );
        }

    protected:
        static constexpr inline HistogramID _char2ID( char a )
        {
            assert( a >= ReducedAlphabet.front());
            return HistogramID( a - ReducedAlphabet.front());
        }

        static constexpr inline char _id2Char( HistogramID id )
        {
            assert( id <= 128 );
            return char( id + ReducedAlphabet.front());
        }


        static HistogramID _sequence2ID( const std::string_view seq,
                                         HistogramID init = 0 )
        {
            HistogramID code = init;
            for (char c : seq)
                code = code * StatesN + _char2ID( c );
            return code;
        }

        static std::string _id2Sequence( HistogramID id, const size_t size, std::string &&acc = "" )
        {
            if ( acc.size() == size ) return acc;
            else return _id2Sequence( id / StatesN, size, _id2Char( id % StatesN ) + acc );
        }


    protected:
        std::unordered_map<Order, std::unordered_map<HistogramID, Histogram >> _histograms;

    };

}
#endif //MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
