//
// Created by asem on 14/09/18.
//

#include "Selection.hpp"

namespace MC
{
Selection union_( const MC::Selection &s1 , const MC::Selection &s2 )
{
    MC::Selection _union;
    std::set<MC::Order> orders;
    for ( auto&[order , _] : s1 ) orders.insert( order );
    for ( auto&[order , _] : s2 ) orders.insert( order );

    for ( auto order : orders )
    {
        auto ids1It = s1.find( order );
        auto ids2It = s2.find( order );
        if ( ids1It != s1.cend() || ids2It != s2.cend())
        {
            auto &result = _union[order];
            if ( ids1It != s1.cend() && ids2It != s2.cend())
                set_union( ids1It->second.cbegin() , ids1It->second.cend() ,
                           ids2It->second.cbegin() , ids2It->second.cend() ,
                           inserter( result , result.begin()));
            else if ( ids1It != s1.cend())
                result = ids1It->second;
            else
                result = ids2It->second;
        }
    }

    return _union;
}

Selection union_( const std::vector<MC::Selection> &sets )
{
    MC::Selection scannedKernels;
    for ( const auto &selection : sets )
    {
        scannedKernels = union_( scannedKernels , selection );
    }
    return scannedKernels;
}

SelectionFlat intersection2( const MC::Selection &s1 , const MC::Selection &s2 )
{
    MC::SelectionFlat sInt;
    for ( auto &[order , ids1] : s1 )
    {
        std::vector<MC::HistogramID> intersect;
        if ( auto ids2It = s2.find( order ); ids2It != s2.cend())
        {
            const auto &ids2 = ids2It->second;
            set_intersection( ids1.cbegin() , ids1.cend() , ids2.cbegin() , ids2.cend() ,
                              back_inserter( intersect ));
        }
        if ( !intersect.empty()) sInt[order] = move( intersect );
    }
    return sInt;
}

Selection intersection( MC::Selection &&s1 , const MC::Selection &s2 ) noexcept
{
    for ( auto &[order , ids1] : s1 )
    {
        std::set<MC::HistogramID> intersect;
        if ( auto ids2It = s2.find( order ); ids2It != s2.cend())
        {
            const auto &ids2 = ids2It->second;
            set_intersection( ids1.cbegin() , ids1.cend() , ids2.cbegin() , ids2.cend() ,
                              inserter( intersect , intersect.end()));
        }
        if ( intersect.empty())
            s1.erase( order );
        else s1[order] = move( intersect );
    }
    return s1;
}

Selection intersection( const MC::Selection &s1 , const MC::Selection &s2 ) noexcept
{
    MC::Selection _intersection;
    std::set<MC::Order> orders;
    for ( auto&[order , _] : s1 ) orders.insert( order );
    for ( auto&[order , _] : s2 ) orders.insert( order );
    for ( auto order : orders )
    {
        try
        {
            auto &ids1 = s1.at( order );
            auto &ids2 = s2.at( order );
            auto &result = _intersection[order];
            set_intersection( ids1.cbegin() , ids1.cend() , ids2.cbegin() , ids2.cend() ,
                              inserter( result , result.end()));
        } catch ( const std::out_of_range & )
        {}
    }
    return _intersection;
}

Selection intersection( const std::vector<MC::Selection> sets , std::optional<double> minCoverage = std::nullopt )
{
    const size_t k = sets.size();
    if ( minCoverage && minCoverage == 0.0 )
    {
        return union_( sets );
    }
    if ( minCoverage && minCoverage > 0.0 )
    {
        const MC::Selection scannedKernels = union_( sets );
        MC::Selection result;
        for ( const auto &[order , ids] : scannedKernels )
        {
            for ( auto id : ids )
            {
                auto shared = count_if( cbegin( sets ) , cend( sets ) ,
                                        [order , id]( const auto &set )
                                        {
                                            const auto &isoKernels = set.at( order );
                                            return isoKernels.find( id ) != isoKernels.cend();
                                        } );
                if ( shared >= minCoverage.value() * k )
                {
                    result[order].insert( id );
                }
            }
        }
        return result;
    } else
    {
        MC::Selection result = sets.front();
        for ( auto i = 1; i < sets.size(); ++i )
        {
            result = intersection( result , sets[i] );
        }
        return result;
    }
}

size_t selectionSize( const Selection &s1 )
{
    size_t sum = 0;
    for ( auto &[order , ids] : s1 )
        sum += ids.size();
    return sum;
}

}