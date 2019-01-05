//
// Created by asem on 01/01/19.
//

#ifndef MARKOVIAN_FEATURES_POLYMORPHICMC_HPP
#define MARKOVIAN_FEATURES_POLYMORPHICMC_HPP

#include "MC.hpp"

namespace MC
{

template< typename AAGrouping >
class PolymorphicMC : public MC<AAGrouping::StatesN>
{
    static constexpr auto States = AAGrouping::StatesN;
    using Base = AbstractMC<States>;

public:
    virtual ~PolymorphicMC() = default;

    double probability( std::string_view polymorphicContext , char polymorphicState ) const override
    {
        if ( polymorphicContext.size() > this->getOrder())
        {
            polymorphicContext.remove_prefix( polymorphicContext.size() - this->getOrder());
        }

        return LabeledEntry::polymorphicSummer<AAGrouping>(
                polymorphicContext , polymorphicState ,
                [this]( std::string_view context , char state )
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
                } );
    }

protected:
    void _incrementInstance( std::string_view context , char state ) override
    {
        LabeledEntry::polymorphicApply<AAGrouping>(
                context , state ,
                [this]( std::string_view context ,
                        char state )
                {
                    auto order = context.size();
                    auto id = Base::_sequence2ID( context );
                    auto c = Base::_char2ID( state );
                    this->_histograms[order][id].increment( c );
                } );
    }

    void _countInstance( std::string_view sequence ) override
    {
        for ( auto a : sequence )
        {
            LabeledEntry::polymorphicApply<AAGrouping>( a , [this]( char state )
            {
                auto c = Base::_char2ID( state );
                this->_histograms[0][0].increment( c );
            } );
        }


        for ( Order distance = 1; distance <= this->getOrder(); ++distance )
            for ( auto i = 0; i < sequence.size() - distance; ++i )
                _incrementInstance( sequence.substr( i , distance ) ,
                                    sequence[i + distance] );
    }
};

}
#endif //MARKOVIAN_FEATURES_POLYMORPHICMC_HPP