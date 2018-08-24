//
// Created by asem on 05/08/18.
//

#ifndef MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP
#define MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP

#include "ConfusionMatrix.hpp"
#include <fcntl.h>

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

    double mcc( size_t k  ) const
    {
        return _statistics.at( k ).mcc( );
    }

    template<typename Function>
    std::pair< double , double > averagingFunction( const Function &fn ) const
    {
        std::vector< double > vals;
        for( auto i = 0; i < _k; ++i )
            vals.push_back( fn( i ));

        double sum = std::accumulate( vals.cbegin(),vals.cend(), double(0));
        double mean = sum / _k;
        double sDev = std::accumulate( vals.cbegin(),vals.cend(),double(0),
                                       [mean]( double acc , double val  ){
            return acc + (val - mean)*(val - mean);
        });
        return {mean,sDev};
    }

    std::pair< double , double > averageAccuracy() const
    {
        return averagingFunction( [this]( size_t k ) {
            return averageAccuracy( k );
        } );
    }

    std::pair< double , double > overallAccuracy() const
    {
        return averagingFunction( [this]( size_t k ) {
            return overallAccuracy( k );
        } );
    }

    std::pair< double , double > microPrecision() const
    {
        return averagingFunction( [this]( size_t k ) {
            return microPrecision( k );
        } );
    }

    std::pair< double , double > microRecall() const
    {
        return averagingFunction( [this]( size_t k ) {
            return microRecall( k );
        } );
    }


    std::pair< double , double > microFScore( double beta = 1 ) const
    {
        return averagingFunction( [this]( size_t k ) {
            return microFScore( k );
        } );
    }

    std::pair< double , double > macroPrecision() const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroPrecision( k );
        } );
    }

    std::pair< double , double > macroRecall() const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroRecall( k );
        } );
    }

    std::pair< double , double > macroFScore( double beta = 1 ) const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroFScore( k );
        } );
    }

    std::pair< double , double > mcc( ) const
    {
        return averagingFunction( [this]( size_t k ) {
            return mcc( k );
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
        printRow( "MCC (multiclass)", mcc());

    }

    template<size_t indentation = 0>
    void printReport( size_t k ) const
    {
        _statistics.at( k ). template printReport<indentation>();
    }

private:
    template<size_t indentation, size_t col1Width = 50>
    static auto _printRowFunction()
    {
        return [=]( const char *col1, std::pair<double,double> &&col2 ) {
            constexpr const char *fmtSpec = "{:<{}}{:<{}}:{:.4f}  (±{:.3f})\n";
            fmt::print( fmtSpec, "", indentation, col1, col1Width, col2.first  , col2.second );
        };
    }

private:
    size_t _k;
    std::vector<ConfusionMatrix<Label>> _statistics;
};


#endif //MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP
