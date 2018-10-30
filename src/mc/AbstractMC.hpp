//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
#define MARKOVIAN_FEATURES_MARKOVCHAINS_HPP

#include "common.hpp"
#include "aminoacids_grouping.hpp"
#include "Histogram.hpp"
#include "MCDefs.h"
#include "Selection.hpp"

namespace MC {

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
        using BackboneProfile = std::unique_ptr<AbstractMC>;
        using BackboneProfiles = std::map<std::string_view, std::unique_ptr<AbstractMC >>;
    public:

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


        template<typename Sequence>
        static inline bool isReducedSequences( const std::vector<Sequence> &sequences )
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

        void train( std::string_view sequences )
        {
            assert( isReducedSequence( sequences ));
            _countInstance( sequences );
            _normalize();
        }

        void train( const std::vector<std::string> &sequences )
        {
            assert( isReducedSequences( sequences ));
            for (const auto &s : sequences)
                _countInstance( s );
            _normalize();
        }


        void train( const std::vector<std::string_view> &sequences )
        {
            assert( isReducedSequences( sequences ));
            for (const auto &s : sequences)
                _countInstance( s );
            _normalize();
        }

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
        virtual void _countInstance( std::string_view sequence ) = 0;

        void _normalize()
        {
            for (auto &[order, isoHistograms] : _histograms)
                for (auto &[contextId, histogram] : isoHistograms)
                    histogram.normalize();
        }

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

    public:

        std::unordered_map<size_t, double>
        extractSparsedFlatFeatures( const Selection &select ) const noexcept
        {
            std::unordered_map<size_t, double> features;

            for (auto[order, id] : LazySelectionsIntersection::intersection(
                    featureSpace( _histograms ), select ))
            {
                auto &histogram = _histograms.at( order ).at( id );
                size_t offset = order * id * StatesN;
                for (auto i = 0; i < StatesN; ++i)
                    features[offset + i] = histogram[i];
            }
            return features;
        }

//        static HeteroHistograms difference

        std::vector<double> extractFlatFeatureVector(
                const Selection &select, double missingVals = 0 ) const noexcept
        {
            std::vector<double> features;

            features.reserve(
                    std::accumulate( std::cbegin( select ), std::cend( select ), size_t( 0 ),
                                     [&]( size_t acc, const auto &pair ) {
                                         return acc + pair.second.size() * StatesN;
                                     } ));

            for (auto &[order, ids] : select)
            {
                if ( auto isoHistogramsIt = _histograms.find( order ); isoHistogramsIt != _histograms.cend())
                {
                    auto &isoHistograms = isoHistogramsIt->second;
                    for (auto id : ids)
                    {
                        if ( auto histogramIt = isoHistograms.find( id ); histogramIt != isoHistograms.cend())
                            features.insert( std::end( features ), std::cbegin( histogramIt->second ),
                                             std::cend( histogramIt->second ));
                        else
                            features.insert( std::end( features ), StatesN, missingVals );

                    }
                } else
                {
                    for (auto id : ids)
                        features.insert( std::end( features ), StatesN, missingVals );
                }
            }
            return features;
        }


        static Selection featureSpace( const HeteroHistograms &profile )
        {
            Selection allFeatureSpace;
            for (const auto &[order, isoHistograms] : profile)
                for (const auto &[id, histogram] : isoHistograms)
                    allFeatureSpace[order].insert( id );
            return allFeatureSpace;
        }

        static Selection populationFeatureSpace( const BackboneProfiles &profiles )
        {
            std::unordered_map<Order, std::set<HistogramID >> allFeatureSpace;
            for (const auto &[cluster, profile] : profiles)
                if ( *profile )
                    for (const auto &[order, isoHistograms] : profile->histograms().get())
                        for (const auto &[id, histogram] : isoHistograms)
                            allFeatureSpace[order].insert( id );
            return allFeatureSpace;
        }

        static Selection jointFeatures( const BackboneProfiles &profiles,
                                        const std::unordered_map<Order, std::set<HistogramID >> &allFeatures,
                                        std::optional<double> minSharedPercentage = std::nullopt )
        {
            Selection joint;
            if ( minSharedPercentage )
            {
                assert( minSharedPercentage > 0.0 );

                const size_t minShared = size_t( profiles.size() * minSharedPercentage.value());
                for (auto &[order, isoFeatures] : allFeatures)
                    for (const auto id : isoFeatures)
                    {
                        auto shared = std::count_if( std::cbegin( profiles ), std::cend( profiles ),
                                                     [order, id]( const auto &p ) {
                                                         const auto histogram = p.second->histogram( order, id );
                                                         return histogram.has_value();
                                                     } );
                        if ( shared >= minShared )
                            joint[order].insert( id );

                    }
            } else
            {
                for (auto &[order, isoFeatures] : allFeatures)
                    for (const auto id : isoFeatures)
                    {
                        bool isJoint = std::all_of( std::cbegin( profiles ), std::cend( profiles ),
                                                    [order, id]( const auto &p ) {
                                                        const auto histogram = p.second->histogram( order, id );
                                                        return histogram.has_value();
                                                    } );
                        if ( isJoint )
                            joint[order].insert( id );

                    }
            }
            return joint;
        }

        operator bool() const
        {
            return !_histograms.empty();
        }

        void filter( const Selection &select ) noexcept
        {
            using LazyIntersection = LazySelectionsIntersection;

            HeteroHistograms selectedHistograms;
            for (auto[order, id] : LazyIntersection::intersection( select, featureSpace( _histograms )))
                selectedHistograms[order][id] = std::move( _histograms.at( order ).at( id ));

            _histograms = std::move( selectedHistograms );
        }


        template<typename ModelGenerator>
        static BackboneProfiles
        train( const std::map<std::string_view, std::vector<std::string >> &training,
               ModelGenerator trainer,
               std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
        {
            BackboneProfiles trainedProfiles;
            for (const auto &[label, sequences] : training)
            {
                trainedProfiles.emplace( label, trainer( sequences, selection ));
            }
            return trainedProfiles;
        }

        template<typename ModelGenerator>
        BackboneProfiles
        static backgroundProfiles( const std::map<std::string_view, std::vector<std::string >> &trainingSequences,
                                   ModelGenerator modelTrainer,
                                   const Selection &selection )
        {
            BackboneProfiles background;
            for (auto &[label, _] : trainingSequences)
            {
                std::vector<std::string_view> backgroundSequences;
                for (auto&[bgLabel, bgSequences] : trainingSequences)
                {
                    if ( bgLabel == label ) continue;
                    for (auto &s : bgSequences)
                        backgroundSequences.push_back( s );
                }
                background.emplace( label, modelTrainer( backgroundSequences, selection ));
            }
            return background;
        }

        template<typename ModelGenerator>
        BackboneProfile
        static backgroundProfile( const std::map<std::string_view, std::vector<std::string >> &trainingSequences,
                                  ModelGenerator modelTrainer,
                                  std::optional<std::reference_wrapper<const Selection>> selection )
        {
            BackboneProfiles background;
            std::vector<std::string_view> backgroundSequences;
            for (auto &[label, seqs] : trainingSequences)
                for (auto &s : seqs)
                    backgroundSequences.push_back( s );

            return modelTrainer( backgroundSequences, selection );
        }

        template<typename ModelGenerator>
        static std::map<std::string, std::vector<HeteroHistograms> >
        trainIndividuals( const std::map<std::string_view, std::vector<std::string >> &training,
                          ModelGenerator trainer,
                          std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
        {
            std::map<std::string_view, std::vector<HeteroHistograms  >> trainedHistograms;
            for (const auto &[label, sequences] : training)
            {
                auto &_trainedHistograms = trainedHistograms[label];
                for (const auto &seq : sequences)
                {
                    auto model = trainer( seq, selection );
                    if ( *model ) _trainedHistograms.emplace_back( std::move( model->convertToHistograms()));
                }
            }
            return trainedHistograms;
        }


        template<typename ModelGenerator>
        static std::pair<Selection, BackboneProfiles>
        filterJointKernels( const std::map<std::string_view, std::vector<std::string >> &trainingClusters,
                            ModelGenerator trainer,
                            double minSharedPercentage = 0.75 )
        {
            assert( minSharedPercentage > 0 && minSharedPercentage <= 1 );
            return filterJointKernels( train( trainingClusters, trainer ), minSharedPercentage );
        }

        static size_t size( const Selection &features )
        {
            return std::accumulate( std::cbegin( features ), std::cend( features ), size_t( 0 ),
                                    []( size_t s, const auto &p ) {
                                        return s + p.second.size();
                                    } );
        }


        static std::pair<Selection, BackboneProfiles>
        filterJointKernels( BackboneProfiles &&profiles, double minSharedPercentage = 0.75 )
        {
            assert( minSharedPercentage >= 0 && minSharedPercentage <= 1 );
            const size_t k = profiles.size();

            auto allFeatures = populationFeatureSpace( profiles );
            auto commonFeatures = jointFeatures( profiles, allFeatures, minSharedPercentage );

            profiles = filter( std::forward<BackboneProfiles>( profiles ), commonFeatures );

            return std::make_pair( std::move( commonFeatures ), std::forward<BackboneProfiles>( profiles ));
        }


        static BackboneProfiles
        filter( BackboneProfiles &&profiles,
                const Selection &selection )
        {
            BackboneProfiles filteredProfiles;
            for (auto &[cluster, profile] : profiles)
            {
                if ( profile->filter( selection ); *profile )
                    filteredProfiles.emplace( cluster, std::move( profile ));
            }
            return filteredProfiles;
        }


        template<typename ModelGenerator>
        static Selection
        withinJointAllUnionKernels( const std::map<std::string_view, std::vector<std::string>> &trainingClusters,
                                    ModelGenerator trainer,
                                    double withinCoverage = 0.5 )
        {
            const size_t k = trainingClusters.size();
            std::vector<Selection> withinKernels;
            for (const auto &[clusterName, sequences] : trainingClusters)
            {
                std::vector<Selection> clusterJointKernels;
                Selection allKernels;
                for (const auto &s : sequences)
                {
                    if ( auto model = trainer( s ); *model )
                        clusterJointKernels.emplace_back( featureSpace( model->histograms().get()));
                }
                withinKernels.emplace_back( intersection( clusterJointKernels, withinCoverage ));
            }
            return union_( withinKernels );
        }
    };


    template<typename Grouping>
    class ModelGenerator
    {
    private:
        using Abstract = AbstractMC<Grouping>;
        using HeteroHistograms = typename Abstract::HeteroHistograms;
        using ConfiguredModelFunction =std::function<std::unique_ptr<Abstract>( void )>;
        const ConfiguredModelFunction _modelFunction;

        explicit ModelGenerator( const ConfiguredModelFunction modelFunction )
                : _modelFunction( modelFunction )
        {}

    public:

        template<typename Model, class... Args>
        static ModelGenerator create( Args &&... args )
        {
            return ModelGenerator(
                    [=]() {
                        return std::unique_ptr<Abstract>( new Model( args... ));
                    } );
        }


        template<typename SequenceData>
        std::unique_ptr<AbstractMC<Grouping>> operator()(
                SequenceData &&sequences ) const
        {
            auto model = _modelFunction();
            model->train( std::forward<SequenceData>( sequences ));
            return model;
        }

        template<typename SequenceData>
        std::unique_ptr<AbstractMC<Grouping>> operator()(
                SequenceData &&sequences, std::optional<std::reference_wrapper<const Selection >> selection ) const
        {
            if ( selection )
            {
                auto model = _modelFunction();
                model->train( std::forward<SequenceData>( sequences ));
                model->filter( selection->get());
                if ( *model ) return std::move( model );
                else return nullptr;
            } else return this->operator()( sequences );
        }

        template<typename SequenceData>
        HeteroHistograms histograms(
                SequenceData &&sequences ) const
        {
            auto model = _modelFunction();
            model->train( std::forward<SequenceData>( sequences ));
            return model->convertToHistograms();
        }
    };
}
#endif //MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
