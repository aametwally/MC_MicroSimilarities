//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
#define MARKOVIAN_FEATURES_MARKOVCHAINS_HPP

#include "common.hpp"
#include "AAGrouping.hpp"
#include "Histogram.hpp"
#include "MCDefs.h"
#include "SparseTransitionMatrix.hpp"
#include "Selection.hpp"
#include "LabeledEntry.hpp"

namespace MC
{

template < size_t States >
class AbstractMC
{
public:

    static constexpr std::array<char , States> ReducedAlphabet = reducedAlphabet<States>();

    using Histogram = buffers::Histogram<States>;
    using Buffer =  typename Histogram::Buffer;
    using BufferIterator =  typename Buffer::iterator;
    using BufferConstIterator =  typename Buffer::const_iterator;

public:
    static constexpr double PseudoCounts = double( 0.1 ) / (States + eps);
    using TransitionMatrices2D = SparseTransitionMatrix2D<States , Histogram , Order , HistogramID>;
    using TransitionMatrices1D = SparseTransitionMatrix1D<States , Histogram , HistogramID>;

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
        return _histograms( order ).has_value();
    }

    bool contains( Order order , HistogramID id ) const
    {
        return _histograms( order , id ).has_value();
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
        assert( LabeledEntry::isReducedSequence<States>( sequence ));
        _countInstance( sequence );
        _normalize();
    }

    void train( const std::vector<std::string> &sequences )
    {
        assert( LabeledEntry::isReducedSequences<States>( sequences ));
        for ( const auto &s : sequences )
            _countInstance( s );
        _normalize();
    }


    void train( const std::vector<std::string_view> &sequences )
    {
        assert( LabeledEntry::isReducedSequences<States>( sequences ));
        for ( const auto &s : sequences )
            _countInstance( s );
        _normalize();
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
    double propensity( std::string_view query , double acc = 0 ) const
    {
        if ( !query.empty())
        {
            char state = query.back();
            query.remove_suffix( 1 ); // now query := context
            return propensity( query , acc + transitionalPropensity( query , state ));
        } else return acc;
    }

    virtual double probability( char a ) const
    {
        if ( LabeledEntry::isPolymorphicReducedAA( a ))
        {
            return 1;
        } else
        {
            if ( auto value = this->_histograms( 0 , 0 , _char2ID( a )); value )
            {
                return value.value();
            } else return 0;
        }
    }

    std::reference_wrapper<const SparseTransitionMatrix2D<States>> histograms() const
    {
        return _histograms;
    }

    SparseTransitionMatrix2D<States> stealHistograms()
    {
        return std::move( _histograms );
    }

    auto histograms( Order distance ) const
    {
        return _histograms( distance );
    }

    std::optional<std::reference_wrapper<const Histogram>> histogram( Order distance , HistogramID id ) const
    {
        return _histograms( distance , id );
    }

    template < typename Histograms >
    void setHistograms( Histograms &&histograms )
    {
        _histograms = std::forward<Histograms>( histograms );
    }

protected:
    virtual void _countInstance( std::string_view sequence ) = 0;

    virtual void _normalize()
    {
        _histograms.forEach( []( Order , HistogramID , Histogram &histogram )
                             {
                                 histogram.normalize();
                             } );
    }

    static constexpr inline HistogramID _char2ID( char a )
    {
        assert( LabeledEntry::isReducedAA<States>( a ));
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
            code = code * States + _char2ID( c );
        return code;
    }

    static std::string _id2Sequence( HistogramID id , const size_t size , std::string &&acc = "" )
    {
        if ( acc.size() == size ) return acc;
        else return _id2Sequence( id / States , size , _id2Char( id % States ) + acc );
    }


protected:
    TransitionMatrices2D _histograms;
    const Order _order;

public:
    std::vector<double> extractFlatFeatureVector(
            const Selection &select , double missingVals = 0 ) const noexcept
    {
        std::vector<double> features;
        features.reserve( size( select ) * States );

        for ( auto &[order , ids] : select )
        {
            if ( auto isoHistogramsOpt = _histograms( order ); isoHistogramsOpt )
            {
                auto &isoHistograms = isoHistogramsOpt.value();
                for ( auto id : ids )
                {
                    if ( auto histogramOpt = isoHistograms( id ); histogramOpt )
                    {
                        features.insert( std::end( features ) , std::cbegin( histogramOpt->get()) ,
                                         std::cend( histogramOpt->get()));
                    } else
                    {
                        features.insert( std::end( features ) , States , missingVals );
                    }
                }
            } else
            {
                for ( auto id : ids )
                    features.insert( std::end( features ) , States , missingVals );
            }
        }
        return features;
    }

    std::vector<double> extractFlatFeatureVector(
            const AbstractMC &reference , double missingVals = 0 ) const noexcept
    {
        std::vector<double> features;
        features.reserve( reference.histogramsCount() * States );

        for ( auto &[order , ids] : reference._histograms )
        {
            if ( auto isoHistogramsOpt = _histograms( order ); isoHistogramsOpt )
            {
                auto &isoHistograms = isoHistogramsOpt.value();
                for ( auto &[id , hist] : ids )
                {
                    if ( auto histogramOpt = isoHistograms( id ); histogramOpt )
                    {
                        features.insert( std::end( features ) , std::cbegin( histogramOpt->get()) ,
                                         std::cend( histogramOpt->get()));
                    } else
                    {
                        features.insert( std::end( features ) , States , missingVals );
                    }
                }
            } else
            {
                for ( auto &[id , hist] : ids )
                    features.insert( std::end( features ) , States , missingVals );
            }
        }
        return features;
    }

    static Selection featureSpace( const TransitionMatrices2D &sparseTransitionMatrices )
    {
        Selection allFeatureSpace;
        for ( const auto &[order , isoHistograms] : sparseTransitionMatrices )
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

    explicit operator bool() const
    {
        return !_histograms.empty();
    }

    void filter( const Selection &select ) noexcept
    {
        using LazyIntersection = LazySelectionsIntersection;

        TransitionMatrices2D selectedHistograms;
        for ( auto[order , id] : LazyIntersection::intersection( select , featureSpace( _histograms )))
            selectedHistograms.swap( order , id )( _histograms( order , id )->get());

        _histograms.swap( selectedHistograms );
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


template < size_t States >
class ModelGenerator
{
private:
    using Abstract = AbstractMC<States>;
    using TransitionMatrices2D = typename Abstract::TransitionMatrices2D;
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
    std::unique_ptr<AbstractMC<States>> operator()(
            SequenceData &&sequences ) const
    {
        auto model = _modelFunction();
        model->train( std::forward<SequenceData>( sequences ));
        return model;
    }

    template < typename SequenceData >
    std::unique_ptr<AbstractMC<States>> operator()(
            SequenceData &&sequences ,
            std::optional<std::reference_wrapper<const Selection >> selection ) const
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
    TransitionMatrices2D histograms(
            SequenceData &&sequences ) const
    {
        auto model = _modelFunction();
        model->train( std::forward<SequenceData>( sequences ));
        return model->stealHistograms();
    }
};
}
#endif //MARKOVIAN_FEATURES_MARKOVCHAINS_HPP
