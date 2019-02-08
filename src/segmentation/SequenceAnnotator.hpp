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
        explicit Segment(
                std::string_view subsequence,
                size_t label,
                double score
        )
                : _subsequence( subsequence ), _label( label ),
                  _score( score )
        {}

        inline std::string_view getSubsequence() const
        {
            return _subsequence;
        }

        inline size_t size() const
        {
            return _subsequence.size();
        }

        inline bool empty() const
        {
            return _subsequence.empty();
        }

        inline size_t getLabel() const
        {
            return _label;
        }

        inline double getScore() const
        {
            return _score;
        }

    private:
        std::string_view _subsequence;
        size_t _label;
        double _score;
    };

    std::string toString( std::string_view prefix = "" ) const
    {
        std::stringstream ss;
        ss << prefix;
        std::vector<std::string> segmented;

        for (const auto &segment : _segments)
            segmented.push_back( io::join2string(
                    std::vector<size_t>( segment.size(), segment.getLabel()), "." ));

        ss << io::join( segmented, "|" );
        return std::move( ss.str());
    }

    std::string toStringCompact( std::string_view prefix = "" ) const
    {
        std::stringstream ss;
        ss << prefix;

        for (const auto &segment: _segments)
        {
            ss << fmt::format( "[C{},L={},S={:.2f}]",
                               segment.getLabel(), segment.size(), segment.getScore());
        }
        return ss.str();
    }

    static std::string toString(
            const std::vector<SequenceAnnotation> &annotations,
            std::string_view prefix = ""
    )
    {
        std::stringstream ss;
        double max = -inf;
        for (auto annotation : annotations)
        {
            max = std::max( max, annotation.getScore());
        }

        for (const auto &annotation : annotations)
        {
            auto score = annotation.getScore();
            char flag = (score == max) ? '*' : ' ';
            ss << fmt::format( "{},{}ΣS={},{}\n", prefix, flag, score, annotation.toStringCompact());
        }

        return ss.str();
    }

    size_t size() const
    {
        return std::accumulate( _segments.cbegin(), _segments.cend(), size_t( 0 ),
                                [](
                                        size_t acc,
                                        const auto &s
                                ) {
                                    return acc + s.size();
                                } );
    }

    const std::vector<Segment> &getSegments() const
    {
        return _segments;
    }

    void addSegment(
            std::string_view seq,
            size_t label,
            double score
    )
    {
        _segments.emplace_back( seq, label, score );
    }

    double getScore() const
    {
        return std::accumulate( _segments.cbegin(), _segments.cend(), 0.0, [](
                double acc,
                auto &seg
        ) {
            return acc + seg.getScore();
        } );
    }

private:
    std::vector<Segment> _segments;
};

class SequenceAnnotator
{
    static constexpr double inf = std::numeric_limits<double>::infinity();

public:
    using ScoreFunction = std::function<double( std::string_view )>;

    explicit SequenceAnnotator(
            std::string_view sequence,
            const std::vector<ScoreFunction> &scoreFunctions
    )
            : _sequence( sequence ),
              _scoreFunctions( scoreFunctions )
    {}

    struct Node
    {
        explicit Node( size_t labels )
                : _lanes( labels, -inf )
        {}

        inline void reset( double val = 0 )
        {
            for (auto &v : _lanes) v = val;
        }

        inline std::pair<size_t, double> getMaxLane() const
        {
            assert( !_lanes.empty());
            auto maxIt = std::max_element( _lanes.cbegin(), _lanes.cend());
            return std::make_pair( maxIt - _lanes.cbegin(), *maxIt );
        }

        inline Node &operator+=( const std::vector<double> &rhs )
        {
            assert( _lanes.size() == rhs.size());

            for (auto i = 0; i < _lanes.size(); ++i)
                _lanes[i] += rhs[i];

            return *this;
        }

        inline Node &operator=( const std::vector<double> &rhs )
        {
            assert( _lanes.size() == rhs.size());

            for (auto i = 0; i < _lanes.size(); ++i)
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
        explicit BacktraceGraph(
                size_t rows,
                size_t columns
        )
        {
            _paths = BacktraceBuffer( rows, BacktraceRow( columns, horizontal ));
            assert( nRows() > 0 );
            assert( nColumns() > 0 );
        }

        inline size_t nRows() const
        {
            return _paths.size();
        }

        void addRow()
        {
            _paths.push_back( BacktraceRow( nColumns(), horizontal ));
        }

        inline size_t nColumns() const
        {
            return _paths.front().size();
        }

        inline void setHorizontal(
                size_t row,
                size_t column
        )
        {
            assert( row < _paths.size() && column < _paths.front().size());
            _paths[row][column] = horizontal;
        }

        inline void setVertical(
                size_t row,
                size_t column
        )
        {
            assert( row < _paths.size() && column < _paths.front().size());
            _paths[row][column] = vertical;
        }

        inline bool isHorizontal(
                size_t row,
                size_t column
        ) const
        {
            assert( row < _paths.size() && column < _paths.front().size());
            return _paths[row][column] == horizontal;
        }

        inline bool isVertical(
                size_t row,
                size_t column
        ) const
        {
            return !isHorizontal( row, column );
        }

        std::list<std::pair<size_t, size_t >> getBacktrace( std::optional<int> row = std::nullopt ) const
        {
            std::list<std::pair<size_t, size_t >> horizontalRanges;
            std::pair<size_t, size_t> currentPath;

            currentPath.second = nColumns() - 1;

            for (int _row = row.value_or( nRows() - 1 ); _row >= 0; --_row)
            {
                for (int column = currentPath.second; column >= 0; --column)
                {
                    if ( isVertical( _row, column ))
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
        return _scoreFunctions.size();
    }

    std::vector<SequenceAnnotation> annotate( size_t maxSegments = 0 ) const
    {
        if ( maxSegments == 0 ) return adaptiveAnnotation();
        else
        {
            BacktraceGraph backtrace( maxSegments, nColumns());
            std::unordered_map<size_t, std::unordered_map<size_t, std::pair<size_t, double> >> labels;

            std::vector<Node> currentLine = runFirstLine( backtrace );
            labels[0][nColumns() - 1] = currentLine.back().getMaxLane();

            for (size_t row = 1; row < maxSegments; ++row)
            {
                currentLine = runLine( row, std::move( currentLine ), backtrace, labels );
                labels[row][nColumns() - 1] = currentLine.back().getMaxLane();
            }

            return makeAnnotations( backtrace, labels );
        }
    }

    std::vector<SequenceAnnotation> adaptiveAnnotation() const
    {
        BacktraceGraph backtrace( 1, nColumns());
        std::unordered_map<size_t, std::unordered_map<size_t, std::pair<size_t, double> >> labels;

        std::vector<Node> currentLine = runFirstLine( backtrace );
        labels[0][nColumns() - 1] = currentLine.back().getMaxLane();

        double oldScore = -inf;
        double newScore = currentLine.back().getMaxLane().second;
        for (size_t row = 1; newScore > oldScore; ++row)
        {
            oldScore = newScore;
            backtrace.addRow();
            currentLine = runLine( row, std::move( currentLine ), backtrace, labels );
            labels[row][nColumns() - 1] = currentLine.back().getMaxLane();
            newScore = currentLine.back().getMaxLane().second;
        }

        return makeAnnotations( backtrace, labels );
    }

protected:
    std::vector<double> _scores( std::string_view subsequence ) const
    {
        std::vector<double> scores;
        scores.reserve( _scoreFunctions.size());
        for (auto fn : _scoreFunctions)
            scores.push_back( fn( subsequence ));
        return scores;
    }

    std::vector<Node> runFirstLine( BacktraceGraph &backtrace ) const
    {
        std::vector<Node> forwardNodes( nColumns(), Node( nLabels()));

        forwardNodes.front().reset();

        for (auto i = 1; i < nColumns(); ++i)
            forwardNodes[i] = forwardNodes[i - 1] + _scores( _sequence.substr( 0, i ));

        for (auto i = 0; i < nColumns(); ++i)
            backtrace.setHorizontal( 0, i );

        return forwardNodes;
    }

    std::vector<Node> runLine(
            size_t row,
            std::vector<Node> &&currentLine,
            BacktraceGraph &backtrace,
            std::unordered_map<size_t, std::unordered_map<size_t, std::pair<size_t, double> >> &labels
    ) const
    {
        assert( row > 0 );
        const size_t maxSegments = backtrace.nRows();
        const size_t len = backtrace.nColumns();
        constexpr size_t standardSegmentLength = 10;

        Node up = currentLine.front();
        size_t lastSplit = 0;
        for (auto column = 0; column < nColumns() - 1; ++column)
        {
            auto &left = currentLine[column];
            auto[maxVLane, maxVLaneValue] = up.getMaxLane();
            auto[maxHLane, maxHLaneValue] = left.getMaxLane();

            up = currentLine[column + 1];

            if ( maxVLaneValue >= maxHLaneValue )
            {
                labels[row - 1][column] = std::make_pair( maxVLane, maxVLaneValue );
                backtrace.setVertical( row, column );
                currentLine[column].reset( maxVLaneValue );
                lastSplit = column;
            } else
            {
                backtrace.setHorizontal( row, column );
            }

            auto currentSubsequence = _sequence.substr( lastSplit, column + 1 );

            currentLine[column + 1] = currentLine[column] + _scores( currentSubsequence );
        }

        auto[maxVLane, maxVLaneValue] = up.getMaxLane();
        auto[maxHLane, maxHLaneValue] = currentLine.back().getMaxLane();

        if ( maxVLaneValue >= maxHLaneValue )
        {
            labels[row - 1][nColumns() - 1] = std::make_pair( maxVLane, maxVLaneValue );
            backtrace.setVertical( row, nColumns() - 1 );
            currentLine[nColumns() - 1].reset( maxVLaneValue );
        } else
            backtrace.setHorizontal( row, nColumns() - 1 );

        backtrace.setVertical( row, 0 );
        return currentLine;
    }

    std::vector<SequenceAnnotation> makeAnnotations(
            const BacktraceGraph &backtrace,
            const std::unordered_map<size_t, std::unordered_map<size_t, std::pair<size_t, double> >> &labels
    ) const
    {
        std::vector<SequenceAnnotation> annotations;

        for (auto i = 0; i < backtrace.nRows(); ++i)
        {
            const std::list<std::pair<size_t, size_t >> ranges = backtrace.getBacktrace( i );
            assert( ranges.size() == i + 1 );

            SequenceAnnotation annotation;

            size_t row = 0;
            double lastScore = 0;
            for (auto[first, last] : ranges)
            {
                assert( last >= first );
                if ( last - first > 0 )
                {
                    auto[label, score] = labels.at( row ).at( last );

                    annotation.addSegment( _sequence.substr( first, last - first ),
                                           label, score - lastScore );
                    lastScore = score;
                }
                ++row;
            }

            annotations.emplace_back( std::move( annotation ));
        }
        return annotations;
    }


private:
    const std::string_view _sequence;
    const std::vector<ScoreFunction> _scoreFunctions;
};


#endif //MARKOVIAN_FEATURES_SEQUENCESEGMENTATION_H
