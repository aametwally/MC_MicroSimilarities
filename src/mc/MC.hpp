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
    using TransitionMatrices2D  = typename Base::TransitionMatrices2D;

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
        this->_histograms.swap( mE._histograms );
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
            if ( auto value = this->_histograms( distance , id , stateID ); value )
            {
                return value.value();
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
            this->_histograms.increment( order , id , Base::PseudoCounts )( c );
        }
    }

    void _countInstance( std::string_view sequence ) override
    {
        for ( auto a : sequence )
        {
            if ( !LabeledEntry::isPolymorphicReducedAA( a ))
            {
                auto c = Base::_char2ID( a );
                this->_histograms.increment( 0 , 0 , Base::PseudoCounts )( c );
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
