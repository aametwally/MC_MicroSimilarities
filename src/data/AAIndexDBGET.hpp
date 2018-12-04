//
// Created by asem on 30/11/18.
//

#ifndef MARKOVIAN_FEATURES_AAINDEXDBGET_H
#define MARKOVIAN_FEATURES_AAINDEXDBGET_H

#include "common.hpp"
#include "AAIndex1Data.h"
#include "AAGrouping.hpp"

/**
 * @brief https://www.genome.jp/aaindex/aaindex_help.html
 *
 */

namespace aaindex
{
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

struct AAIndex
{
    static constexpr int16_t INVALID_AA_INDEX = -1;

    static constexpr auto AAOrders = aaOrders_STRICT( INVALID_AA_INDEX );

    explicit AAIndex( const std::map<char , double> &index )
            : _index( asArray( index )) {}

    AAIndex() = default;

    virtual ~AAIndex() = default;

    inline double operator()( char aa ) const
    {
        auto idx = AAOrders.at( aa - CHAR_RANGE.first );
        if ( idx >= 0 )
        {
            return _index.at( idx );
        } else if ( auto it = POLYMORPHIC_AA.find( aa ); it != POLYMORPHIC_AA.cend())
        {
            auto aas = it->second;
            double sum = std::accumulate( aas.cbegin() , aas.cend() , double( 0 ) ,
                                          [this]( double acc , char aa )
                                          {
                                              return acc + this->operator()( aa );
                                          } );
            return sum / aas.length();
        } else
        {
            return nan;
        }
    }

    inline bool hasMissingValues() const
    {
        return std::any_of( _index.cbegin() , _index.cend() , []( double v ) { return v == nan; } );
    }

    inline std::vector<double> sequence2TimeSeries( std::string_view sequence ) const
    {
        std::vector<double> series;
        series.reserve( sequence.length());
        std::transform( sequence.cbegin() , sequence.cend() , std::back_inserter( series ) ,
                        [this]( char aa ) { return this->operator()( aa ); } );
        return series;
    }

    static std::array<double , AA_COUNT > asArray( const std::map<char , double> &index )
    {
        assert( index.size() == AA_COUNT );
        auto arr = std::array<double , AA_COUNT >();
        std::fill( arr.begin() , arr.end() , 0 );
        for ( auto[aa , value] : index )
        {
            assert( aa - CHAR_RANGE.first >= 0 );
            assert( AAOrders.at( aa - CHAR_RANGE.first ) >= 0 );
            arr[AAOrders.at( aa )] = value;
        }
        return arr;
    }

protected:
    std::array<double , AA_COUNT> _index;
};

struct NormalizedAAIndex : public AAIndex
{
    explicit NormalizedAAIndex( const std::map<char , double> &index )
    {
        std::tie( _mean , _std , _index ) = normalizeIndex( index );
    }

    inline double getMean() const
    {
        return _mean;
    }

    inline double getSigma() const
    {
        return _std;
    }

    static std::tuple<double , double , std::array<double , AA_COUNT >>
    normalizeIndex( const std::map<char , double> &index )
    {
        std::array<double , AA_COUNT> normalizedIndex = asArray( index );
        auto n = AA_COUNT - std::count( normalizedIndex.cbegin() , normalizedIndex.cend() , nan );
        double sum = std::accumulate( normalizedIndex.cbegin() , normalizedIndex.cend() , double( 0 ) ,
                                      []( double acc , double v )
                                      {
                                          if ( std::isnan( v ))
                                              return acc;
                                          else return acc + v;
                                      } );

        double mean = sum / n;
        double var = std::accumulate( normalizedIndex.cbegin() , normalizedIndex.cend() , double( 0 ) ,
                                      [=]( double acc , double v )
                                      {
                                          if ( std::isnan( v ))
                                              return acc;
                                          else return acc + (v - mean) * (v - mean);
                                      } ) / n;
        double sigma = std::sqrt( var );
        for ( auto &v : normalizedIndex )
            v = (v - mean) / sigma;

        return std::make_tuple( mean , sigma , normalizedIndex );
    }

protected:
    double _mean;
    double _std;
};


class AAIndex1
{
public:
    struct MetaData
    {
        std::string accessionNumber;
        std::string dataDescription;
        std::string PMID;
        std::string authors;
        std::string articleTitle;
    };

    template < typename MetaDataType , typename CorrelationsType >
    explicit AAIndex1( MetaDataType &&mData ,
                       const std::map<char , double> &index ,
                       CorrelationsType &&correlations )
            : _metaData( std::forward<MetaData>( mData )) ,
              _index( index ) ,
              _normalizedIndex( index ) ,
              _correlations( std::forward<CorrelationsType>( correlations )) {}

    inline const std::string &getAccessionNumber() const
    {
        return _metaData.accessionNumber;
    }

    inline const std::string &getDataDescription() const
    {
        return _metaData.dataDescription;
    }

    inline const std::string &getPMID() const
    {
        return _metaData.PMID;
    }

    inline const std::string &getAuthors() const
    {
        return _metaData.authors;
    }

    inline const std::string &getArticleTitle() const
    {
        return _metaData.articleTitle;
    }

    inline bool hasMissingValues() const
    {
        return _index.hasMissingValues();
    }

    inline std::vector<double> sequence2NormalizedTimeSeries( std::string_view sequence ) const
    {
        return _normalizedIndex.sequence2TimeSeries( sequence );
    }

    inline std::vector<double> sequence2TimeSeries( std::string_view sequence ) const
    {
        return _index.sequence2TimeSeries( sequence );
    }

    inline auto index() const
    {
        return [this]( char aa ) { return _index( aa ); };
    }

    inline auto normalizedIndex() const
    {
        return [this]( char aa ) { return _normalizedIndex( aa ); };
    }

    inline const std::map<std::string , double> &getCorrelations() const
    {
        return _correlations;
    }

private:
    const MetaData _metaData;
    const std::map<std::string , double> _correlations;
    const AAIndex _index;
    const NormalizedAAIndex _normalizedIndex;
};


std::string extractContent( std::string_view block , std::string_view flag )
{

    auto lines = io::split( block , "\n" );
    auto first = std::find_if( lines.begin() , lines.end() ,
                               [=]( const std::string &line )
                               {
                                   return line.substr( 0 , flag.size()) == flag;
                               } );
    auto last = std::find_if( first + 1 , lines.end() ,
                              []( const std::string &line )
                              {
                                  return line.front() != ' ';
                              } );
    auto tokens = io::split( *first , " " );
    *first = io::join( tokens.cbegin() + 1 , tokens.cend() , " " );
    return io::join( first , last , "\n" );
}

AAIndex1::MetaData extractMetaData( std::string_view block )
{
    AAIndex1::MetaData mData;
    mData.accessionNumber = extractContent( block , ACESSION_NUMBER_FLAG );
    mData.articleTitle = extractContent( block , ARTICLE_TITLE_FLAG );
    mData.authors = extractContent( block , AUTHORS_FLAG );
    mData.dataDescription = extractContent( block , DATA_DESCRIPTION_FLAG );
    mData.PMID = extractContent( block , PMID_FLAG );
    return mData;
}

std::map<char , double> extractIndex( std::string block )
{
    auto indexBlock = extractContent( block , AMINOACID_INDEX_FLAG );
    auto lines = io::split( indexBlock , "\n" );

    auto tokens = io::split( lines.front());
    assert( tokens.size() == 10 );
    std::string aaRow1( 10 , 0 );
    std::string aaRow2( 10 , 0 );
    for ( auto it = tokens.cbegin(); it != tokens.cend(); ++it )
    {
        auto order = std::distance( tokens.cbegin() , it );
        auto aas = io::split( *it , '/' );
        aaRow1.at( order ) = aas.front().front();
        aaRow2.at( order ) = aas.at( 1 ).front();
    }
    std::stringstream indexStream1( lines.at( 1 ));
    std::stringstream indexStream2( lines.at( 2 ));

    std::map<char , double> index;
    for ( auto i = 0; i < 10; ++i )
    {
        std::string valueStr;
        indexStream1 >> valueStr;
        if ( valueStr == MISSING_VALUE )
            index[aaRow1.at( i )] = nan;
        else index[aaRow1.at( i )] = std::stod( valueStr );

        indexStream2 >> valueStr;
        if ( valueStr == MISSING_VALUE )
            index[aaRow2.at( i )] = nan;
        else index[aaRow2.at( i )] = std::stod( valueStr );

    }
    assert( index.size() == 20 );
    index['O'] = nan;
    index['U'] = nan;
    return index;
}

std::map<std::string , double> extractCorrelations( std::string_view block )
{
    auto indexBlock = extractContent( block , CORRELATED_ACESSION_NUMBERS_FLAG );
    auto lines = io::split( indexBlock , "\n" );
    std::map<std::string , double> correlations;
    for ( auto &line : lines )
    {
        std::stringstream ss( line );
        while ( !ss.eof())
        {
            std::string accession;
            double value = 0;
            ss >> accession >> value;
            assert( !std::isnan( value ));
            correlations.emplace( accession , value );
        }
    }
    return correlations;
}

std::map<std::string , AAIndex1> extractAAIndices()
{
    auto blocks = io::split( io::trim_copy( AAINDEX1_DATA ) , BLOCK_DELIMETER );
    std::map<std::string , AAIndex1> indices;
    for ( auto &block : blocks )
    {
        io::trim( block );
        auto mData = extractMetaData( block );
        auto index = extractIndex( block );
        auto correlations = extractCorrelations( block );
        std::string key = mData.accessionNumber;
        indices.emplace( std::piecewise_construct ,
                         std::forward_as_tuple( std::move( key )) ,
                         std::forward_as_tuple( std::move( mData ) ,
                                                std::move( index ) ,
                                                std::move( correlations )));
    }
    return indices;
}
}

#endif //MARKOVIAN_FEATURES_AAINDEXDBGET_H
