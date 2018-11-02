//
// Created by asem on 30/10/18.
//

#ifndef MARKOVIAN_FEATURES_SEQUENCESEGMENTATION_H
#define MARKOVIAN_FEATURES_SEQUENCESEGMENTATION_H

#include "common.hpp"


struct SequenceAnnotation
{
    std::string_view segment;
    size_t label = 0;
};

class SequenceAnnotator
{
    static constexpr double inf = std::numeric_limits<double>::infinity();

public:
    explicit SequenceAnnotator( std::string_view sequence , std::vector<std::vector<double >> &&scores )
            : _sequence( sequence ) , _scores( std::move( scores ))
    {
        assert( _sequence.size() == _scores.size());
        size_t k = _scores.front().size();
        assert( std::all_of( _scores.cbegin() , _scores.cend() , [k]( const auto &v ) { return v.size() == k; } ));
    }

    struct ForwardEdge
    {
        explicit ForwardEdge( size_t labels )
                : _lanes( labels , -inf ) {}

        inline void reset( double val = 0 )
        {
            for ( auto &v : _lanes ) v = val;
        }


        inline std::pair<size_t , double> getMaxLane() const
        {
            auto maxIt = std::max_element( _lanes.cbegin() , _lanes.cend());
            return std::make_pair( maxIt - _lanes.cbegin() , *maxIt );
        }

        inline ForwardEdge &operator+=( const std::vector<double> &rhs )
        {
            assert( _lanes.size() == rhs.size());

            for ( auto i = 0; i < _lanes.size(); ++i )
                _lanes[i] += rhs[i];

            return *this;
        }

        inline ForwardEdge &operator=( const std::vector<double> &rhs )
        {
            assert( _lanes.size() == rhs.size());

            for ( auto i = 0; i < _lanes.size(); ++i )
                _lanes[i] == rhs[i];

            return *this;
        }

        inline ForwardEdge operator+( const std::vector<double> &rhs ) const
        {
            ForwardEdge copy = *this;
            return copy += rhs;
        }

    private:
        std::vector<double> _lanes;
    };

    class BacktraceGraph
    {
    public:
        explicit BacktraceGraph( size_t columns , size_t maxSegments )
        {
            _paths = std::vector<std::vector<bool >>( maxSegments , std::vector<bool>( columns , false ));
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
            _paths[row][column] = true;
        }

        inline void setVertical( size_t row , size_t column )
        {
            assert( row < _paths.size() && column < _paths.front().size());
            _paths[row][column] = false;
        }

        inline bool isHorizontal( size_t row , size_t column ) const
        {
            assert( row < _paths.size() && column < _paths.front().size());
            return _paths[row][column];
        }

        inline bool isVertical( size_t row , size_t column ) const
        {
            return !isHorizontal( row , column );
        }

        std::list<std::pair<size_t , size_t >> getBacktrace() const
        {
            std::list<std::pair<size_t , size_t >> horizontalPaths;
            std::pair<size_t , size_t> currentPath;
            currentPath.second = nColumns() - 1;
            for ( int row = nRows() - 1; row >= 0; --row )
            {
                for ( int column = currentPath.second; column >= 0; --column )
                {
                    if ( isVertical( row , column ))
                    {
                        currentPath.first = column;
                        horizontalPaths.push_front( currentPath );
                        currentPath.second = column;
                        break;
                    }
                }
            }
            return horizontalPaths;
        }

    private:
        std::vector<std::vector<bool >> _paths;
    };

    inline size_t nColumns() const
    {
        return _sequence.size();
    }

    inline size_t nLabels() const
    {
        return _scores.front().size();
    }

    std::vector<SequenceAnnotation> annotate( size_t maxSegments ) const
    {
        BacktraceGraph backtrace( nColumns() , maxSegments );
        std::unordered_map<size_t , std::unordered_map<size_t , size_t >> verticalMultiplexes;

        std::vector<ForwardEdge> currentLine = runFirstLine( backtrace );


        for ( size_t row = 1; row <= maxSegments; ++row )
        {
            currentLine = runLine( row , std::move( currentLine ) , backtrace , verticalMultiplexes );
        }

        return makeAnnotations( backtrace , verticalMultiplexes );
    }


protected:
    std::vector<ForwardEdge> runFirstLine( BacktraceGraph &backtrace ) const
    {
        std::vector<ForwardEdge> forwardEdges( nColumns() , ForwardEdge( nLabels()));

        forwardEdges.front() = _scores.front();

        for ( auto i = 1; i < nColumns(); ++i )
            forwardEdges[i] = forwardEdges[i - 1] + _scores[i];

        for ( auto i = 0; i < nColumns(); ++i )
            backtrace.setHorizontal( 0 , i );

        return forwardEdges;
    }

    std::vector<ForwardEdge> runLine(
            size_t row ,
            std::vector<ForwardEdge> &&currentLine ,
            BacktraceGraph &backtrace ,
            std::unordered_map<size_t , std::unordered_map<size_t , size_t >> &vMultiplexes ) const
    {
        assert( row > 0 );

        ForwardEdge vertical = currentLine.front();
        for ( auto column = 0; column < nColumns() - 1; ++column )
        {
            auto[maxVLane , maxVLaneValue] = vertical.getMaxLane();
            auto[maxHLane , maxHLaneValue] = currentLine[column].getMaxLane();

            vertical = currentLine[column + 1];

            if ( maxVLaneValue >= maxHLaneValue )
            {
                vMultiplexes[row - 1][column] = maxVLane;
                backtrace.setVertical( row , column );
                currentLine[column + 1].reset( maxVLaneValue );
                currentLine[column + 1] += _scores[column + 1];

            } else
            {
                backtrace.setHorizontal( row , column );
                currentLine[column + 1] = currentLine[column] + _scores[column + 1];
            }
        }

        auto[maxVLane , maxVLaneValue] = vertical.getMaxLane();
        auto[maxHLane , maxHLaneValue] = currentLine.back().getMaxLane();

        if ( maxVLaneValue >= maxHLaneValue )
        {
            vMultiplexes[row - 1][nColumns() - 1] = maxVLane;
            backtrace.setVertical( row , nColumns() - 1 );
        } else
            backtrace.setHorizontal( row , nColumns() - 1 );

        return currentLine;
    }

    std::vector<SequenceAnnotation> makeAnnotations(
            const BacktraceGraph &backtrace ,
            const std::unordered_map<size_t , std::unordered_map<size_t , size_t >> &vMultiplexes ) const
    {
        const std::list<std::pair<size_t , size_t >> hRoads = backtrace.getBacktrace();
        std::vector<SequenceAnnotation> annotations;
        for ( auto[first , last] : hRoads )
        {
            size_t row = annotations.size();
            SequenceAnnotation annotation;
            annotation.segment = _sequence.substr( first , last - first );
            annotation.label = vMultiplexes.at( row ).at( last );
            annotations.push_back( annotation );
        }
        return annotations;
    }

private:
    std::string_view _sequence;
    const std::vector<std::vector<double >> _scores;
};


#endif //MARKOVIAN_FEATURES_SEQUENCESEGMENTATION_H
