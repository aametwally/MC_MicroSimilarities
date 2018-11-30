//
// Created by asem on 30/11/18.
//

#ifndef MARKOVIAN_FEATURES_AAINDEXDBGET_H
#define MARKOVIAN_FEATURES_AAINDEXDBGET_H

#include "common.hpp"
#include "AAIndex1Data.h"

/**
 * @brief https://www.genome.jp/aaindex/aaindex_help.html
 *
 */

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
        std::string comments;
    };

    template < typename MetaDataType , typename IndexType , typename CorrelationsType >
    explicit AAIndex1( MetaDataType &&mData ,
                       IndexType &&index ,
                       CorrelationsType &&correlations )
            : _metaData( std::forward<MetaData>( mData )) ,
              _index( std::forward<IndexType>( index )) ,
              _correlations( std::forward<CorrelationsType>( correlations )) {}

    const std::string &getAccessionNumber() const
    {
        return _metaData.accessionNumber;
    }

    const std::string &getDataDescription() const
    {
        return _metaData.dataDescription;
    }

    const std::string &getPMID() const
    {
        return _metaData.PMID;
    }

    const std::string &getAuthors() const
    {
        return _metaData.authors;
    }

    const std::string &getArticleTitle() const
    {
        return _metaData.articleTitle;
    }

    const std::string &getComments() const
    {
        return _metaData.comments;
    }

    const std::unordered_map<char , double> &getIndex() const
    {
        return _index;
    }

    const std::map<std::string , double> &getCorrelations() const
    {
        return _correlations;
    }

private:
    const MetaData _metaData;
    const std::unordered_map<char , double> _index;
    const std::map<std::string , double> _correlations;
};

std::map< std::string , AAIndex1 > extractAAIndices()
{
//    auto items = io::split( AAIn)
}

#endif //MARKOVIAN_FEATURES_AAINDEXDBGET_H
