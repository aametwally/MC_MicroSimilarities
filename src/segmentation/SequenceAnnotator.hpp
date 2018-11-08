//
// Created by asem on 30/10/18.
//

#ifndef MARKOVIAN_FEATURES_SEQUENCESEGMENTATION_H
#define MARKOVIAN_FEATURES_SEQUENCESEGMENTATION_H

#include "common.hpp"


struct SequenceAnnotation
{
    static constexpr double inf = std::numeric_limits<double>::infinity();

    struct Segment
    {
        explicit Segment( std::string_view subsequence , size_t label , double score )
                : subsequence( subsequence ) , label( label ) , score( score ) {}

        std::string_view subsequence;
        size_t label;
        double score;
    };

    std::string toString( std::string_view prefix = "" ) const
    {
        std::stringstream ss;
        ss << prefix;
        std::vector<std::string> segmented;

        for ( const auto &segment : segments )
            segmented.push_back( io::join2string(
                    std::vector<size_t>( segment.subsequence.size() , segment.label ) , "." ));

        ss << io::join( segmented , "|" );
        return std::move( ss.str());
    }

    std::string toStringCompact( std::string_view prefix = "" ) const
    {
        std::stringstream ss;
        ss << prefix;
        for ( const auto &segment: segments )
            ss << fmt::format( "[C{}:L={}]" , segment.label , segment.subsequence.size());
        return ss.str();
    }

    std::pair<size_t , double> topLabel() const
    {
        std::map<size_t , double> acc;
        for ( auto &s : segments )
        {
            assert( !s.subsequence.empty());
            const size_t n = s.subsequence.size();
            acc[s.label] += ( s.score );
        }
        auto max = std::max_element( acc.cbegin() , acc.cend() , []( const auto &p1 , const auto &p2 )
        {
            return p1.second < p2.second;
        } );
        return *max;
    }

    static std::string toString( const std::vector<SequenceAnnotation> &annotations ,
                                 std::string_view prefix = "" )
    {
        std::stringstream ss;
        double max = -inf;
        for ( auto annotation : annotations )
        {
            max = std::max( max , annotation.score );
        }

        for ( const auto &annotation : annotations )
        {
            auto[label , score] = annotation.topLabel();
            char flag = (annotation.score == max) ? '*' : ' ';
            ss << fmt::format( "{}\n" ,
                               annotation.toStringCompact( fmt::format( "{},top={},{}" , prefix , label , flag )));
        }
        return ss.str();
    }

    std::vector<Segment> segments;
    double score;
};

class SequenceAnnotator
{
    static constexpr double inf = std::numeric_limits<double>::infinity();

public:
    explicit SequenceAnnotator( std::string_view sequence , std::vector<std::vector<double >> &&scores )
            : _sequence( sequence ) , _scores( center( transpose( std::move( scores ))))
    {
        assert( _sequence.size() == _scores.size());
        size_t k = _scores.front().size();
        assert( std::all_of( _scores.cbegin() , _scores.cend() , [k]( const auto &v ) { return v.size() == k; } ));
    }

    static std::vector<std::vector<double>> center( std::vector<std::vector<double >> &&scores )
    {
        for( auto &v : scores )
        {
            double sum = std::accumulate( v.cbegin() , v.cend() , double(0));
            for( auto &s : v ) s -= sum;
        }
        return scores;
    }

    static std::vector<std::vector<double >> transpose( std::vector<std::vector<double >> &&scores )
    {
        const auto len = scores.front().size();
        const auto k = scores.size();
        std::vector<std::vector<double >> scoresT( len , std::vector<double>( k , 0 ));
        for ( auto i = 0; i < len; ++i )
            for ( auto j = 0; j < k; ++j )
                scoresT[i][j] = scores[j][i];
        return scoresT;
    }

    struct Node
    {
        explicit Node( size_t labels )
                : _lanes( labels , -inf ) {}

        inline void reset( double val = 0 )
        {
            for ( auto &v : _lanes ) v = val;
        }


        inline std::pair<size_t , double> getMaxLane() const
        {
            assert( !_lanes.empty());
            auto maxIt = std::max_element( _lanes.cbegin() , _lanes.cend());
            return std::make_pair( maxIt - _lanes.cbegin() , *maxIt );
        }

        inline Node &operator+=( const std::vector<double> &rhs )
        {
            assert( _lanes.size() == rhs.size());

            for ( auto i = 0; i < _lanes.size(); ++i )
                _lanes[i] += rhs[i];

            return *this;
        }

        inline Node &operator=( const std::vector<double> &rhs )
        {
            assert( _lanes.size() == rhs.size());

            for ( auto i = 0; i < _lanes.size(); ++i )
                _lanes[i] = rhs[i];

            return *this;
        }

        inline Node operator+( const std::vector<double> &rhs ) const
        {
            Node copy = *this;
            return copy += rhs;
        }

    private:
        std::vector<double> _lanes;
    };

    class BacktraceGraph
    {
        using BacktraceLabel = bool;
        static constexpr BacktraceLabel horizontal = false;
        static constexpr BacktraceLabel vertical = true;
        using BacktraceRow = std::vector<BacktraceLabel>;
        using BacktraceBuffer = std::vector<BacktraceRow>;
    public:
        explicit BacktraceGraph( size_t rows , size_t columns )
        {
            _paths = BacktraceBuffer( rows , BacktraceRow( columns , horizontal ));
            assert( nRows() > 0 );
            assert( nColumns() > 0 );
        }

        inline size_t nRows() const
        {
            return _paths.size();
        }

        inline size_t nColumns() const
        {
            return _paths.front().size();
        }

        inline void setHorizontal( size_t row , size_t column )
        {
            assert( row < _paths.size() && column < _paths.front().size());
            _paths[row][column] = horizontal;
        }

        inline void setVertical( size_t row , size_t column )
        {
            assert( row < _paths.size() && column < _paths.front().size());
            _paths[row][column] = vertical;
        }

        inline bool isHorizontal( size_t row , size_t column ) const
        {
            assert( row < _paths.size() && column < _paths.front().size());
            return _paths[row][column] == horizontal;
        }

        inline bool isVertical( size_t row , size_t column ) const
        {
            return !isHorizontal( row , column );
        }

        std::list<std::pair<size_t , size_t >> getBacktrace( std::optional<int> row = std::nullopt ) const
        {
            std::list<std::pair<size_t , size_t >> horizontalRanges;
            std::pair<size_t , size_t> currentPath;

            currentPath.second = nColumns() - 1;

            for ( int _row = row.value_or( nRows() - 1 ); _row >= 0; --_row )
            {
                for ( int column = currentPath.second; column >= 0; --column )
                {
                    if ( isVertical( _row , column ))
                    {
                        currentPath.first = column;
                        horizontalRanges.push_front( currentPath );
                        currentPath.second = column;
                        break;
                    }
                }
            }

            currentPath.first = 0;
            horizontalRanges.push_front( currentPath );
            return horizontalRanges;
        }

    private:
        BacktraceBuffer _paths;
    };

    inline size_t nColumns() const
    {
        return _sequence.size() + 1;
    }

    inline size_t nLabels() const
    {
        return _scores.front().size();
    }

    std::vector<SequenceAnnotation> annotate( size_t maxSegments ) const
    {
        BacktraceGraph backtrace( maxSegments , nColumns());
        std::unordered_map<size_t , std::unordered_map<size_t , std::pair<size_t , double> >> labels;

        std::vector<Node> currentLine = runFirstLine( backtrace );
        labels[0][nColumns() - 1] = currentLine.back().getMaxLane();

        for ( size_t row = 1; row < maxSegments; ++row )
        {
            currentLine = runLine( row , std::move( currentLine ) , backtrace , labels );
            labels[row][nColumns() - 1] = currentLine.back().getMaxLane();
        }

        return makeAnnotations( backtrace , labels );
    }


protected:
    std::vector<Node> runFirstLine( BacktraceGraph &backtrace ) const
    {
        std::vector<Node> forwardEdges( nColumns() , Node( nLabels()));

        forwardEdges.front().reset();

        for ( auto i = 1; i < nColumns(); ++i )
            forwardEdges[i] = forwardEdges[i - 1] + _scores[i - 1];

        for ( auto i = 0; i < nColumns(); ++i )
            backtrace.setHorizontal( 0 , i );

        return forwardEdges;
    }

    std::vector<Node> runLine(
            size_t row ,
            std::vector<Node> &&currentLine ,
            BacktraceGraph &backtrace ,
            std::unordered_map<size_t , std::unordered_map<size_t , std::pair<size_t , double> >> &labels ) const
    {
        assert( row > 0 );
        const size_t maxSegments = backtrace.nRows();
        const size_t len = backtrace.nColumns();
        const double penalty = - std::log( maxSegments );

        Node up = currentLine.front();

        for ( auto column = 0; column < nColumns() - 1; ++column )
        {
            auto &left = currentLine[column];
            auto[maxVLane , maxVLaneValue] = up.getMaxLane();
            auto[maxHLane , maxHLaneValue] = left.getMaxLane();

            up = currentLine[column + 1];

            if ( maxVLaneValue + penalty >= maxHLaneValue )
            {
                labels[row - 1][column] = std::make_pair( maxVLane , maxVLaneValue );
                backtrace.setVertical( row , column );
                currentLine[column].reset( maxVLaneValue + penalty );
            } else
            {
                backtrace.setHorizontal( row , column );
            }
            currentLine[column + 1] = currentLine[column] + _scores[column];
        }

        auto[maxVLane , maxVLaneValue] = up.getMaxLane();
        auto[maxHLane , maxHLaneValue] = currentLine.back().getMaxLane();

        if ( maxVLaneValue + penalty >= maxHLaneValue )
        {
            labels[row - 1][nColumns() - 1] = std::make_pair( maxVLane , maxVLaneValue );
            backtrace.setVertical( row , nColumns() - 1 );
            currentLine[nColumns() - 1].reset( maxVLaneValue + penalty );
        } else
            backtrace.setHorizontal( row , nColumns() - 1 );

        backtrace.setVertical( row , 0 );

        return currentLine;
    }

    std::vector<SequenceAnnotation> makeAnnotations(
            const BacktraceGraph &backtrace ,
            const std::unordered_map<size_t , std::unordered_map<size_t , std::pair<size_t , double> >> &labels ) const
    {
        std::vector<SequenceAnnotation> annotations;

        for ( auto i = 0; i < backtrace.nRows(); ++i )
        {
            const std::list<std::pair<size_t , size_t >> ranges = backtrace.getBacktrace( i );
            assert( ranges.size() == i + 1 );

            SequenceAnnotation annotation;

            size_t row = 0;
            for ( auto[first , last] : ranges )
            {
                assert( last >= first );
                if ( last - first > 0 )
                {
                    auto[label , score] = labels.at( row ).at( last );
                    annotation.segments.emplace_back( _sequence.substr( first , last - first ) , label , score );
                }
                ++row;
            }

            annotation.score = annotation.segments.back().score;
            for ( auto i = 1; i < annotation.segments.size(); ++i )
                annotation.segments[i].score -= annotation.segments[i - 1].score;

            annotations.emplace_back( std::move( annotation ));
        }
        return annotations;
    }

private:
    std::string_view _sequence;
    const std::vector<std::vector<double >> _scores;
};


#endif //MARKOVIAN_FEATURES_SEQUENCESEGMENTATION_H
