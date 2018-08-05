//
// Created by asem on 05/08/18.
//

#ifndef MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP
#define MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP

#include "ConfusionMatrix.hpp"

template<typename Label = std::string>
class CrossValidationStatistics
{
public:
    CrossValidationStatistics( size_t k, const std::set<Label> &labels )
            : _k( k )
    {
        for (auto i = 0; i < k; ++i)
            _statistics.emplace_back( labels );
    }


    void countInstance( size_t k, const Label &prediction, const Label &actual )
    {
        _statistics.at( k ).countInstance( prediction, actual );
    }


    double averageAccuracy( size_t k ) const
    {
        return _statistics.at( k ).averageAccuracy();
    }

    double overallAccuracy( size_t k ) const
    {
        return _statistics.at( k ).overallAccuracy();
    }

    double microPrecision( size_t k ) const
    {
        return _statistics.at( k ).microPrecision();
    }

    double microRecall( size_t k ) const
    {
        return _statistics.at( k ).microRecall();
    }


    double microFScore( size_t k, double beta = 1 ) const
    {
        return _statistics.at( k ).microFScore( beta );
    }

    double macroPrecision( size_t k ) const
    {
        return _statistics.at( k ).macroPrecision();
    }

    double macroRecall( size_t k ) const
    {
        return _statistics.at( k ).macroRecall();
    }

    double macroFScore( size_t k, double beta = 1 ) const
    {
        return _statistics.at( k ).macroFScore( beta );
    }

    template<typename Function>
    double averagingFunction( const Function &fn ) const
    {
        double sum = 0;
        for (auto i = 0; i < _k; ++i)
            sum += fn( i );
        return sum / _k;
    }

    double averageAccuracy() const
    {
        return averagingFunction( [this]( size_t k ) {
            return averageAccuracy( k );
        } );
    }

    double overallAccuracy() const
    {
        return averagingFunction( [this]( size_t k ) {
            return overallAccuracy( k );
        } );
    }

    double microPrecision() const
    {
        return averagingFunction( [this]( size_t k ) {
            return microPrecision( k );
        } );
    }

    double microRecall() const
    {
        return averagingFunction( [this]( size_t k ) {
            return microRecall( k );
        } );
    }


    double microFScore( double beta = 1 ) const
    {
        return averagingFunction( [this]( size_t k ) {
            return microFScore( k );
        } );
    }

    double macroPrecision() const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroPrecision( k );
        } );
    }

    double macroRecall() const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroRecall( k );
        } );
    }

    double macroFScore( double beta = 1 ) const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroFScore( k );
        } );
    }

    template<size_t indentation = 0>
    void printReport() const
    {
        fmt::print( "{:<{}}General Statistics:\n", "", indentation );

        auto printRow = _printRowFunction<indentation + 2>();

        printRow( "Overall Accuracy", overallAccuracy());
        printRow( "Average Accuracy", averageAccuracy());
        printRow( "Macro Precision, Positive Predictive Value (PPV)", macroPrecision());
        printRow( "Micro Precision, Positive Predictive Value (PPV)", microPrecision());
        printRow( "Macro TPR, Recall, Sensitivity", macroRecall());
        printRow( "Micro TPR, Recall, Sensitivity", microRecall());
        printRow( "Macro F1-Score", macroFScore());
        printRow( "Micro F1-Score", microFScore());
    }

private:
    template<size_t indentation, size_t col1Width = 50>
    static auto _printRowFunction()
    {
        return [=]( const char *col1, double col2 ) {
            constexpr const char *fmtSpec = "{:<{}}{:<{}}:{}\n";
            fmt::print( fmtSpec, "", indentation, col1, col1Width, col2 );
        };
    }

private:
    size_t _k;
    std::vector<ConfusionMatrix<Label>> _statistics;
};


#endif //MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP
