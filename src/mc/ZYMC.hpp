//
// Created by asem on 10/09/18.
//

#ifndef MARKOVIAN_FEATURES_ZYMC_HPP
#define MARKOVIAN_FEATURES_ZYMC_HPP

#include "AbstractMC.hpp"
#include "SequenceEntry.hpp"

namespace MC
{
/**
 * @brief ZYMC
 * Zheng Yuan Approximated Higher-order Markov Chains
 * Paper: https://febs.onlinelibrary.wiley.com/doi/pdf/10.1016/S0014-5793%2899%2900506-2
 */
template < size_t States >
class ZYMC : public AbstractMC<States>
{
public:
    using Base = AbstractMC<States>;
    using Histogram = typename Base::Histogram;

    using IsoHistograms = std::unordered_map<HistogramID , Histogram>;
    using HeteroHistograms = std::unordered_map<Order , IsoHistograms>;

public:
    explicit ZYMC( Order order ) : Base( order )
    {
        assert( order >= 1 );
    }

    virtual ~ZYMC() = default;

    template < typename HistogramsCollection >
    explicit ZYMC( Order order , HistogramsCollection &&histograms )
            :  Base( std::forward<HistogramsCollection>( histograms ) , order )
    {
        assert( order >= 1 );
    }

    explicit ZYMC( const std::vector<std::string> &sequences ,
                   Order order ) : Base( order )
    {
        assert( order >= 1 );
        this->train( sequences );
    }


    static constexpr inline HistogramID lowerOrderID( HistogramID id ) { return id / States; }

    inline double pairwiseProbability( char context ,
                                       char state ,
                                       Order distance ) const
    {
        auto c = Base::_char2ID( context );
        auto s = Base::_char2ID( state );
        auto value = this->_histograms( distance , c , s );
        return value.value_or( 0.0 );
    }

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
            double p = 1.0;
            for ( auto i = 0; i < context.size(); ++i )
            {
                auto distance = Order( context.size() - i );
                auto c = context[i];
                p *= pairwiseProbability( c , state , distance );
            }
            return p;
        }
    }

protected:
    virtual void _incrementInstance( std::string_view context ,
                                     char state ,
                                     Order distance )
    {
        assert( context.size() == 1 );

        if ( !LabeledEntry::isPolymorphicReducedSequence<States>( context ) &&
             !LabeledEntry::isPolymorphicReducedAA( state ))
        {
            auto c = Base::_char2ID( context.front());
            auto s = Base::_char2ID( state );
            this->_histograms.increment( distance , c , Base::PseudoCounts )( s );
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

        for ( Order distance = 1; distance <= this->_order; ++distance )
            for ( auto i = 0; i + distance < sequence.size(); ++i )
                _incrementInstance( sequence.substr( i , 1 ) , sequence[i + distance] , distance );
    }
};
}
#endif //MARKOVIAN_FEATURES_ZYMC_HPP
