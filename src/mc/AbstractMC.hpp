//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
#define MARKOVIAN_FEATURES_MARKOVCHAINS_HPP

#include "common.hpp"
#include "AAGrouping.hpp"
#include "Histogram.hpp"
#include "MCDefs.h"
#include "Selection.hpp"
#include "LabeledEntry.hpp"

namespace MC
{

template < typename AAGrouping >
class AbstractMC
{
public:
    static constexpr size_t StatesN = AAGrouping::StatesN;
    static constexpr std::array<char , StatesN> ReducedAlphabet = reducedAlphabet<StatesN>();
    static constexpr std::array<int16_t , 256> ReducedAlphabetIds = reducedAlphabetIds( AAGrouping::Grouping );

    using Histogram = buffers::Histogram<StatesN>;
    using Buffer =  typename Histogram::Buffer;
    using BufferIterator =  typename Buffer::iterator;
    using BufferConstIterator =  typename Buffer::const_iterator;

public:
    using IsoHistograms = std::unordered_map<HistogramID , Histogram>;
    using HeteroHistograms =  std::unordered_map<Order , IsoHistograms>;
    using HeteroHistogramsFeatures = std::unordered_map<Order , std::unordered_map<HistogramID , double>>;
    using BackboneProfile = std::unique_ptr<AbstractMC>;
    using BackboneProfiles = std::map<std::string_view , std::unique_ptr<AbstractMC >>;
public:

public:

    template < typename Histograms >
    AbstractMC( Histograms &&histograms , Order order )
            : _histograms( std::forward<Histograms>( histograms )) , _order( order ) {}


    AbstractMC( Order order ) : _order( order ) {}

    AbstractMC( AbstractMC &&mE ) noexcept
            : _histograms( std::move( mE._histograms )) , _order( mE.getOrder()) {}

    AbstractMC( const AbstractMC &mE ) = default;

    virtual ~AbstractMC() = default;

    AbstractMC &operator=( const AbstractMC &mE )
    {
        assert( _order == mE._order );
        if ( _order != mE._order )
            throw std::runtime_error( "Orders mismatch!" );
        this->_histograms = mE._histograms;
        return *this;
    }

    AbstractMC &operator=( AbstractMC &&mE )
    {
        assert( _order == mE._order );
        if ( _order != mE._order )
            throw std::runtime_error( "Orders mismatch!" );
        this->_histograms = std::move( mE._histograms );
        return *this;
    }

    size_t histogramsCount() const
    {
        size_t sum = 0;
        for ( auto &[order , isoHistograms] : _histograms )
            sum += isoHistograms.size();
        return sum;
    }

    bool contains( Order order ) const
    {
        auto isoHistogramsIt = _histograms.find( order );
        return isoHistogramsIt != _histograms.cend();
    }

    bool contains( Order order , HistogramID id ) const
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
        for ( auto &[order , isoHistograms] : _histograms )
        {
            for ( auto &[id , histogram] : isoHistograms )
            {
                features[order].insert( id );
            }
        }
        return features;
    }

    void train( std::string_view sequence )
    {
        assert( LabeledEntry::isReducedSequence<AAGrouping>( sequence ));
        _countInstance( sequence );
        _normalize();
    }

    void train( const std::vector<std::string> &sequences )
    {
        assert( LabeledEntry::isReducedSequences<AAGrouping>( sequences ));
        for ( const auto &s : sequences )
            _countInstance( s );
        _normalize();
    }


    void train( const std::vector<std::string_view> &sequences )
    {
        assert( LabeledEntry::isReducedSequences<AAGrouping>( sequences ));
        for ( const auto &s : sequences )
            _countInstance( s );
        _normalize();
    }

    inline std::vector<double> compensatedPropensityVector( std::string_view query ) const
    {
        std::vector<double> pVector;
        pVector.reserve( query.size());

        pVector.push_back( probability( query.front()));
        for ( int i = 1; i < query.size(); ++i )
        {
            double p = probability( query.substr( 0 , i ) , query[i] );
            pVector.push_back( std::log( p ) / std::min( i , int( _order )));
        }

        assert( pVector.size() == query.size());
        return pVector;
    }

    inline Order getOrder() const
    {
        return _order;
    }

    virtual double probability( std::string_view , char ) const = 0;

    inline double transitionalPropensity( std::string_view context , char state ) const
    {
        if ( context.empty())
            return std::log2( probability( state ));
        else return std::log2( probability( context , state ));
    }

    /**
     * @brief
     * A tail recursive, markov chain rule.
     * @param query
     * @param acc
     * @return
     */
    inline double propensity( std::string_view query , double acc = 0 ) const
    {
        if ( !query.empty())
        {
            char state = query.back();
            query.remove_suffix( 1 ); // now query := context
            return propensity( query , acc + transitionalPropensity( query , state ));
        } else return acc;
    }

    inline double probability( char a ) const
    {
        return _polymorphicSummer( a , [this]( char state )
        {
            return this->_histograms.at( 0 ).at( 0 ).at( _char2ID( state ));
        } );
    }

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

    std::optional<std::reference_wrapper<const Histogram>> histogram( Order distance , HistogramID id ) const
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

    template < typename Histograms >
    void set( Histograms &&histograms )
    {
        _histograms = std::forward<Histograms>( histograms );
    }

protected:

    static double _polymorphicSummer( char polymorphicState ,
                                      std::function<double( char )> fn ,
                                      double acc = 0 )
    {
        if ( LabeledEntry::isPolymorphicReducedAA( polymorphicState ))
        {
            auto stateMutations = LabeledEntry::generateReducedAAMutations<AAGrouping>( polymorphicState );
            return std::accumulate( stateMutations.cbegin() ,
                                    stateMutations.cend() , acc ,
                                    [fn]( double acc , char stateMutation )
                                    {
                                        return acc + fn( stateMutation );
                                    } );
        } else
        {
            return acc + fn( polymorphicState );
        }
    }

    static double _polymorphicSummer( std::string_view polymorphicContext , char polymorphicState ,
                                      std::function<double( std::string_view , char )> fn ,
                                      double acc = 0 )
    {
        if ( LabeledEntry::isPolymorphicReducedSequence<AAGrouping>( polymorphicContext ))
        {
            auto contextMutations =
                    LabeledEntry::generateReducedPolymorphicSequenceOutcome<AAGrouping>( polymorphicContext );
            return std::accumulate( contextMutations.cbegin() , contextMutations.cend() , acc ,
                                    [=]( double acc , std::string_view contextMutation )
                                    {
                                        return acc + _polymorphicSummer( polymorphicState ,
                                                                         [contextMutation , fn]( char state )
                                                                         {
                                                                             return fn( contextMutation , state );
                                                                         } );
                                    } );
        } else
        {
            return acc + _polymorphicSummer( polymorphicState , [=]( char state )
            {
                return fn( polymorphicContext , state );
            } );
        }
    }

    template < typename AppliedFn >
    static void _polymorphicApply( char polymorphicState , AppliedFn fn )
    {
        if ( LabeledEntry::isPolymorphicReducedAA( polymorphicState ))
        {
            auto stateMutations = LabeledEntry::generateReducedAAMutations<AAGrouping>( polymorphicState );
            std::for_each( stateMutations.cbegin() , stateMutations.cend() , fn );
        } else
        {
            fn( polymorphicState );
        }
    }

    template < typename AppliedFn >
    static void _polymorphicApply( std::string_view polymorphicContext , char polymorphicState , AppliedFn fn )
    {
        if ( LabeledEntry::isPolymorphicReducedSequence<AAGrouping>( polymorphicContext ))
        {
            auto contextMutations = LabeledEntry::generateReducedPolymorphicSequenceOutcome<AAGrouping>(
                    polymorphicContext );
            std::for_each( contextMutations.cbegin() , contextMutations.cend() ,
                           [=]( std::string_view contextMutation )
                           {
                               _polymorphicApply( polymorphicState , [=]( char state )
                               {
                                   fn( contextMutation , state );
                               } );
                           } );
        } else
        {
            _polymorphicApply( polymorphicState , [=]( char state )
            {
                fn( polymorphicContext , state );
            } );
        }
    }

    virtual void _countInstance( std::string_view sequence ) = 0;

    virtual void _normalize()
    {
        for ( auto &[order , isoHistograms] : _histograms )
            for ( auto &[contextId , histogram] : isoHistograms )
                histogram.normalize();
    }

    static constexpr inline HistogramID _char2ID( char a )
    {
        assert( LabeledEntry::isReducedAA<AAGrouping>( a ));
        return HistogramID( a - ReducedAlphabet.front());
    }

    static constexpr inline char _id2Char( HistogramID id )
    {
        assert( id <= 128 );
        return char( id + ReducedAlphabet.front());
    }


    static HistogramID _sequence2ID( const std::string_view seq ,
                                     HistogramID init = 0 )
    {
        HistogramID code = init;
        for ( char c : seq )
            code = code * StatesN + _char2ID( c );
        return code;
    }

    static std::string _id2Sequence( HistogramID id , const size_t size , std::string &&acc = "" )
    {
        if ( acc.size() == size ) return acc;
        else return _id2Sequence( id / StatesN , size , _id2Char( id % StatesN ) + acc );
    }


protected:
    std::unordered_map<Order , std::unordered_map<HistogramID , Histogram >> _histograms;
    const Order _order;
    mutable std::pair<std::string_view , double> _cache;

public:

    std::unordered_map<size_t , double>
    extractSparsedFlatFeatures( const Selection &select ) const noexcept
    {
        std::unordered_map<size_t , double> features;

        for ( auto[order , id] : LazySelectionsIntersection::intersection(
                featureSpace( _histograms ) , select ))
        {
            auto &histogram = _histograms.at( order ).at( id );
            size_t offset = order * id * StatesN;
            for ( auto i = 0; i < StatesN; ++i )
                features[offset + i] = histogram[i];
        }
        return features;
    }

    std::vector<double> extractFlatFeatureVector(
            const Selection &select , double missingVals = 0 ) const noexcept
    {
        std::vector<double> features;

        features.reserve(
                std::accumulate( std::cbegin( select ) , std::cend( select ) , size_t( 0 ) ,
                                 [&]( size_t acc , const auto &pair )
                                 {
                                     return acc + pair.second.size() * StatesN;
                                 } ));

        for ( auto &[order , ids] : select )
        {
            if ( auto isoHistogramsIt = _histograms.find( order ); isoHistogramsIt != _histograms.cend())
            {
                auto &isoHistograms = isoHistogramsIt->second;
                for ( auto id : ids )
                {
                    if ( auto histogramIt = isoHistograms.find( id ); histogramIt != isoHistograms.cend())
                        features.insert( std::end( features ) , std::cbegin( histogramIt->second ) ,
                                         std::cend( histogramIt->second ));
                    else
                        features.insert( std::end( features ) , StatesN , missingVals );

                }
            } else
            {
                for ( auto id : ids )
                    features.insert( std::end( features ) , StatesN , missingVals );
            }
        }
        return features;
    }


    static Selection featureSpace( const HeteroHistograms &profile )
    {
        Selection allFeatureSpace;
        for ( const auto &[order , isoHistograms] : profile )
            for ( const auto &[id , histogram] : isoHistograms )
                allFeatureSpace[order].insert( id );

        return allFeatureSpace;
    }

    static Selection populationFeatureSpace( const BackboneProfiles &profiles )
    {
        std::unordered_map<Order , std::set<HistogramID >> allFeatureSpace;
        for ( const auto &[cluster , profile] : profiles )
            if ( *profile )
                for ( const auto &[order , isoHistograms] : profile->histograms().get())
                    for ( const auto &[id , histogram] : isoHistograms )
                        allFeatureSpace[order].insert( id );
        return allFeatureSpace;
    }

    static Selection jointFeatures( const BackboneProfiles &profiles ,
                                    const std::unordered_map<Order , std::set<HistogramID >> &allFeatures ,
                                    std::optional<double> minSharedPercentage = std::nullopt )
    {
        Selection joint;
        if ( minSharedPercentage )
        {
            assert( minSharedPercentage > 0.0 );

            const size_t minShared = size_t( profiles.size() * minSharedPercentage.value());
            for ( auto &[order , isoFeatures] : allFeatures )
                for ( const auto id : isoFeatures )
                {
                    auto shared = std::count_if( std::cbegin( profiles ) , std::cend( profiles ) ,
                                                 [order , id]( const auto &p )
                                                 {
                                                     const auto histogram = p.second->histogram( order , id );
                                                     return histogram.has_value();
                                                 } );
                    if ( shared >= minShared )
                        joint[order].insert( id );

                }
        } else
        {
            for ( auto &[order , isoFeatures] : allFeatures )
                for ( const auto id : isoFeatures )
                {
                    bool isJoint = std::all_of( std::cbegin( profiles ) , std::cend( profiles ) ,
                                                [order , id]( const auto &p )
                                                {
                                                    const auto histogram = p.second->histogram( order , id );
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
        for ( auto[order , id] : LazyIntersection::intersection( select , featureSpace( _histograms )))
            selectedHistograms[order][id] = std::move( _histograms.at( order ).at( id ));

        _histograms = std::move( selectedHistograms );
    }


    template < typename ModelGenerator , typename Sequence >
    static BackboneProfiles
    train( const std::map<std::string_view , std::vector<Sequence >> &training ,
           ModelGenerator trainer ,
           std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
    {
        BackboneProfiles trainedProfiles;
        for ( const auto &[label , sequences] : training )
        {
            trainedProfiles.emplace( label , trainer( sequences , selection ));
        }
        return trainedProfiles;
    }

    template < typename ModelGenerator >
    BackboneProfiles
    static backgroundProfiles( const std::map<std::string_view , std::vector<std::string >> &trainingSequences ,
                               ModelGenerator modelTrainer ,
                               std::optional<std::reference_wrapper<const Selection> > selection = std::nullopt )
    {
        BackboneProfiles background;
        for ( auto &[label , _] : trainingSequences )
        {
            std::vector<std::string_view> backgroundSequences;
            for ( auto&[bgLabel , bgSequences] : trainingSequences )
            {
                if ( bgLabel != label )
                {
                    for ( auto &s : bgSequences )
                        backgroundSequences.push_back( s );
                }
            }
            background.emplace( label , modelTrainer( backgroundSequences , selection ));
        }
        return background;
    }

    template < typename ModelGenerator >
    BackboneProfiles
    static backgroundProfilesSampled(
            const std::map<std::string_view , std::vector<std::string >> &trainingSequences ,
            ModelGenerator modelTrainer ,
            const Selection &selection )
    {
        BackboneProfiles background;
        auto minimumCluster = std::min_element( trainingSequences.cbegin() , trainingSequences.cend() ,
                                                []( const auto &cluster1 , const auto &cluster2 )
                                                {
                                                    return cluster1.second.size() < cluster2.second.size();
                                                } );
        auto sampleSize = minimumCluster->second.size();

        for ( auto &[label , _] : trainingSequences )
        {
            std::vector<std::string_view> backgroundSequences;
            for ( auto clusterIt = trainingSequences.cbegin(); clusterIt != trainingSequences.cend(); ++clusterIt )
            {
                if ( clusterIt->first == label ) continue;
                if ( clusterIt == minimumCluster )
                {
                    backgroundSequences.insert( backgroundSequences.end() ,
                                                clusterIt->second.cbegin() ,
                                                clusterIt->second.cend());
                } else
                {
                    const auto[subset , rest] = subsetRandomSeparation
                            <std::vector<std::string> , std::vector<std::string_view>>( clusterIt->second ,
                                                                                        sampleSize );
                    backgroundSequences.insert( backgroundSequences.end() ,
                                                subset.cbegin() ,
                                                subset.cend());
                }
            }
            background.emplace( label , modelTrainer( backgroundSequences , selection ));
        }
        return background;
    }

    template < typename ModelGenerator >
    BackboneProfile
    static backgroundProfile( const std::map<std::string_view , std::vector<std::string >> &trainingSequences ,
                              ModelGenerator modelTrainer ,
                              std::optional<std::reference_wrapper<const Selection>> selection )
    {
        BackboneProfiles background;
        std::vector<std::string_view> backgroundSequences;
        for ( auto &[label , seqs] : trainingSequences )
            for ( auto &s : seqs )
                backgroundSequences.push_back( s );

        return modelTrainer( backgroundSequences , selection );
    }

    template < typename ModelGenerator >
    static std::map<std::string , std::vector<HeteroHistograms> >
    trainIndividuals( const std::map<std::string_view , std::vector<std::string >> &training ,
                      ModelGenerator trainer ,
                      std::optional<std::reference_wrapper<const Selection >> selection = std::nullopt )
    {
        std::map<std::string_view , std::vector<HeteroHistograms  >> trainedHistograms;
        for ( const auto &[label , sequences] : training )
        {
            auto &_trainedHistograms = trainedHistograms[label];
            for ( const auto &seq : sequences )
            {
                auto model = trainer( seq , selection );
                if ( *model ) _trainedHistograms.emplace_back( std::move( model->convertToHistograms()));
            }
        }
        return trainedHistograms;
    }


    template < typename ModelGenerator >
    static std::pair<Selection , BackboneProfiles>
    filterJointKernels( const std::map<std::string_view , std::vector<std::string >> &trainingClusters ,
                        ModelGenerator trainer ,
                        double minSharedPercentage = 0.75 )
    {
        assert( minSharedPercentage > 0 && minSharedPercentage <= 1 );
        return filterJointKernels( train( trainingClusters , trainer ) , minSharedPercentage );
    }

    static size_t size( const Selection &features )
    {
        return std::accumulate( std::cbegin( features ) , std::cend( features ) , size_t( 0 ) ,
                                []( size_t s , const auto &p )
                                {
                                    return s + p.second.size();
                                } );
    }


    static std::pair<Selection , BackboneProfiles>
    filterJointKernels( BackboneProfiles &&profiles , double minSharedPercentage = 0.75 )
    {
        assert( minSharedPercentage >= 0 && minSharedPercentage <= 1 );
        const size_t k = profiles.size();

        auto allFeatures = populationFeatureSpace( profiles );
        auto commonFeatures = jointFeatures( profiles , allFeatures , minSharedPercentage );

        profiles = filter( std::forward<BackboneProfiles>( profiles ) , commonFeatures );

        return std::make_pair( std::move( commonFeatures ) , std::forward<BackboneProfiles>( profiles ));
    }


    static BackboneProfiles
    filter( BackboneProfiles &&profiles ,
            const Selection &selection )
    {
        BackboneProfiles filteredProfiles;
        for ( auto &[cluster , profile] : profiles )
        {
            if ( profile->filter( selection ); *profile )
                filteredProfiles.emplace( cluster , std::move( profile ));
        }
        return filteredProfiles;
    }


    template < typename ModelGenerator >
    static Selection
    withinJointAllUnionKernels( const std::map<std::string_view , std::vector<std::string>> &trainingClusters ,
                                ModelGenerator trainer ,
                                double withinCoverage = 0.5 )
    {
        const size_t k = trainingClusters.size();
        std::vector<Selection> withinKernels;
        for ( const auto &[clusterName , sequences] : trainingClusters )
        {
            std::vector<Selection> clusterJointKernels;
            for ( const auto &s : sequences )
            {
                if ( auto model = trainer( s ); *model )
                    clusterJointKernels.emplace_back( featureSpace( model->histograms().get()));
            }
            withinKernels.emplace_back( intersection( clusterJointKernels , withinCoverage ));
        }
        return union_( withinKernels );
    }
};


template < typename Grouping >
class ModelGenerator
{
private:
    using Abstract = AbstractMC<Grouping>;
    using HeteroHistograms = typename Abstract::HeteroHistograms;
    using ConfiguredModelFunction =std::function<std::unique_ptr<Abstract>( void )>;
    const ConfiguredModelFunction _modelFunction;

    explicit ModelGenerator( const ConfiguredModelFunction modelFunction )
            : _modelFunction( modelFunction ) {}

public:

    template < typename Model , class... Args >
    static ModelGenerator create( Args &&... args )
    {
        return ModelGenerator(
                [=]()
                {
                    return std::unique_ptr<Abstract>( new Model( args... ));
                } );
    }


    template < typename SequenceData >
    std::unique_ptr<AbstractMC<Grouping>> operator()(
            SequenceData &&sequences
                                                    ) const
    {
        auto model = _modelFunction();
        model->train( std::forward<SequenceData>( sequences ));
        return model;
    }

    template < typename SequenceData >
    std::unique_ptr<AbstractMC<Grouping>> operator()(
            SequenceData &&sequences , std::optional<std::reference_wrapper<const Selection >>
    selection ) const
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

    template < typename SequenceData >
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
