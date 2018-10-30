//
// Created by asem on 19/08/18.
//

#ifndef MARKOVIAN_FEATURES_FEATURESCOREAUC_HPP
#define MARKOVIAN_FEATURES_FEATURESCOREAUC_HPP

#include "common.hpp"

class FeatureScoreAUC
{
private:
    struct Case
    {
        Case( double score, bool tp ) : _featureScore( score ), _tp( tp )
        {}

        Case( Case &&other ) = default;

        Case( const Case &other ) = default;

        Case &operator=( const Case &other )
        {
            _featureScore = other._featureScore;
            _tp = other._tp;
            return *this;
        }

        inline double score() const
        { return _featureScore; }

        inline bool tp() const
        { return _tp; }

    private:
        double _featureScore;
        bool _tp;
    };

public:
    void record( double score, bool tp )
    {
        _cases.emplace_back( score, tp );
    }

    double auc()
    {

        if ( std::all_of( std::begin( _cases ), std::end( _cases ), []( const Case &c ) {
            return std::isnan( c.score());
        } ))
        {
            fmt::print( "All scores are NaN.\n" );
            return -1;
        }

        auto it = std::partition( std::begin( _cases ), std::end( _cases ), []( const Case &c ) {
            return !std::isnan( c.score());
        } );

        std::sort( std::begin( _cases ), it, []( const Case &c1, const Case &c2 ) {
            return c1.score() > c2.score();
        } );


        auto maxScore = _cases.front().score();
        auto minScoreIt = std::find_if( _cases.crbegin(), _cases.crend(), []( const Case &c1 ) {
            return !std::isnan( c1.score());
        } );
        if ( maxScore == minScoreIt->score())
        {
            fmt::print( "Scores are flat (max-min=0).\n" );
            return -1;
        }

        const auto tpCount = std::count_if( std::begin( _cases ), std::end( _cases ),
                                            []( const Case &c ) { return c.tp(); } );
        const auto fnCount = _cases.size() - tpCount;

        if ( tpCount == 0 )
            return 0.0;
        else if ( fnCount == 0 )
            return 1.0;
        else
        {

            double dTP = 1.0 / tpCount;
            double dFN = 1.0 / fnCount;
            double area = 0;
            double bar = 0;
            for (auto &c : _cases)
            {
                if ( c.tp()) bar += dTP;
                else area += bar * dFN;
            }
            return area;
        }
    }

    size_t n() const
    {
        return _cases.size();
    }

    auto tp() const
    {
        return std::count_if( std::begin( _cases ), std::end( _cases ),
                              []( const Case &c ) { return c.tp(); } );
    }

    auto fn() const
    {
        return std::count_if( std::begin( _cases ), std::end( _cases ),
                              []( const Case &c ) { return !c.tp(); } );
    }

    auto nans() const
    {
        return std::count_if( std::begin( _cases ), std::end( _cases ),
                              []( const Case &c ) { return std::isnan( c.score()); } );
    }

    double max() const
    {
        auto it = std::max_element( std::begin( _cases ), std::end( _cases ),
                                    []( const Case &c1, const Case &c2 ) { return c1.score() < c2.score(); } );
        return it->score();
    }

    double min() const
    {
        auto it = std::min_element( std::begin( _cases ), std::end( _cases ),
                                    []( const Case &c1, const Case &c2 ) { return c1.score() < c2.score(); } );
        return it->score();
    }

    std::pair<double, double> minMax() const
    {
        auto mm = std::minmax_element( std::begin( _cases ), std::end( _cases ),
                                       []( const Case &c1, const Case &c2 ) { return c1.score() < c2.score(); } );
        return {mm.first->score(), mm.second->score()};
    }

    double range() const
    {
        auto mm = minMax();
        return mm.second - mm.first;
    }

    std::string scores2String() const
    {
        std::vector<double> scores;
        std::transform( _cases.cbegin(), _cases.cend(), std::back_inserter( scores ),
                        []( const Case &c ) { return c.score(); } );
        return io::join2string( scores, " " );
    }

    std::string tpfn2String() const
    {
        std::string scores;
        for( auto &c : _cases )
        {
            scores += (c.tp())? "1 ": "0 ";
        }
        return scores;
    }

private:
    std::vector<Case> _cases;
};

#endif //MARKOVIAN_FEATURES_FEATURESCOREAUC_HPP
