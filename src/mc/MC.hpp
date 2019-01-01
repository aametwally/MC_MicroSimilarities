//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_MC_HPP
#define MARKOVIAN_FEATURES_MC_HPP

#include "AbstractMC.hpp"

namespace MC
{

template < size_t States >
class MC : public AbstractMC<States>
{

public:
    using Base = AbstractMC<States>;
    using Histogram = typename Base::Histogram;
    using HeteroHistograms  = typename Base::HeteroHistograms;

public:
    explicit MC( Order order ) : Base( order )
    {
        assert( order >= 1 );
    }

    template < typename HistogramsCollection >
    explicit MC( Order order , HistogramsCollection &&histograms )
            :  Base( std::forward<HistogramsCollection>( histograms ) , order )
    {
        assert( order >= 1 );
    }

    explicit MC( const std::vector<std::string> &sequences ,
                 Order order ) : Base( order )
    {
        assert( order >= 1 );
        this->train( sequences );
    }

    explicit MC( const std::vector<std::string_view> &sequences ,
                 Order order ) : Base( order )
    {
        assert( order >= 1 );
        this->train( sequences );
    }

    MC() = delete;

    virtual ~MC() = default;

    MC( const MC &mE ) = default;

    MC( MC &&mE ) noexcept
            : Base( std::move( mE._histograms ) , mE._order ) {}

    MC &operator=( const MC &mE )
    {
        if ( this->getOrder() != mE.getOrder())
            throw std::runtime_error( "Orders mismatch!" );
        this->_histograms = mE._histograms;
        return *this;
    }

    MC &operator=( MC &&mE )
    {
        if ( this->getOrder() != mE.getOrder())
            throw std::runtime_error( "Orders mismatch!" );
        this->_histograms = std::move( mE._histograms );
        return *this;
    }

    static constexpr inline HistogramID lowerOrderID( HistogramID id ) { return id / States; }


    double probability( std::string_view context , char state ) const override
    {
        if ( context.size() > this->getOrder())
        {
            context.remove_prefix( context.size() - this->getOrder());
        }

        if ( LabeledEntry::isPolymorphicReducedSequence<States>( context ) ||
             LabeledEntry::isPolymorphicReducedAA( state ))
        {
            return 1;
        } else
        {
            auto distance = Order( context.length());
            auto id = Base::_sequence2ID( context );
            auto stateID = Base::_char2ID( state );
            if ( auto isoHistogramsIt = this->_histograms.find( distance );
                    isoHistogramsIt != this->_histograms.cend())
            {
                auto &isoHistograms = isoHistogramsIt->second;
                if ( auto histogramIt = isoHistograms.find( id ); histogramIt !=
                                                                  isoHistograms.cend())
                {
                    auto &histogram = histogramIt->second;
                    return histogram[stateID];
                } else return 0.0;
            } else return 0.0;
        }
    }

protected:
    virtual void _incrementInstance( std::string_view context ,
                                     char state )
    {
        if ( !LabeledEntry::isPolymorphicReducedSequence<States>( context ) &&
             !LabeledEntry::isPolymorphicReducedAA( state ))
        {
            auto order = context.size();
            auto id = Base::_sequence2ID( context );
            auto c = Base::_char2ID( state );
            this->_histograms[order][id].increment( c );
        }
    }

    void _countInstance( std::string_view sequence ) override
    {
        for ( auto a : sequence )
        {
            if ( !LabeledEntry::isPolymorphicReducedAA( a ))
            {
                auto c = Base::_char2ID( a );
                this->_histograms[0][0].increment( c );
            }
        }


        for ( Order distance = 1; distance <= this->getOrder(); ++distance )
            for ( auto i = 0; i < sequence.size() - distance; ++i )
                _incrementInstance( sequence.substr( i , distance ) ,
                                    sequence[i + distance] );
    }
};

}
#endif //MARKOVIAN_FEATURES_MC_HPP
