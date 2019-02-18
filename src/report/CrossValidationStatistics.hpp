//
// Created by asem on 05/08/18.
//

#ifndef MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP
#define MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP

#include "ConfusionMatrix.hpp"
#include <fcntl.h>

template<typename Label = std::string_view>
class CrossValidationStatistics
{
public:
    CrossValidationStatistics(
            size_t k,
            const std::set<Label> &labels
    )
            : _k( k )
    {
        for (auto i = 0; i < k; ++i)
            _statistics.emplace_back( labels );
    }

    CrossValidationStatistics() = default;

    void countInstance(
            size_t k,
            const Label &prediction,
            const Label &actual
    )
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

    double microSpecificity( size_t k ) const
    {
        return _statistics.at( k ).microSpecificity();
    }

    double microFScore(
            size_t k,
            double beta = 1
    ) const
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

    double macroSpecificity( size_t k ) const
    {
        return _statistics.at( k ).macroSpecificity();
    }

    double macroFScore(
            size_t k,
            double beta = 1
    ) const
    {
        return _statistics.at( k ).macroFScore( beta );
    }

    double mcc( size_t k ) const
    {
        return _statistics.at( k ).mcc();
    }

    std::map<Label, std::map<Label, size_t >>
    misclassifications( size_t k ) const
    {
        return _statistics.at( k ).misclassifications();
    }

    template<typename Function>
    std::pair<double, double> averagingFunction( const Function &fn ) const
    {
        std::vector<double> vals;
        for (auto i = 0; i < _k; ++i)
            vals.push_back( fn( i ));

        double sum = std::accumulate( vals.cbegin(), vals.cend(), double( std::numeric_limits<double>::epsilon()));
        double mean = sum / _k;
        double sDev = std::accumulate( vals.cbegin(), vals.cend(), double( 0 ),
                                       [mean](
                                               double acc,
                                               double val
                                       ) {
                                           return acc + (val - mean) * (val - mean);
                                       } );
        return {mean, sDev};
    }

    double accuracy(
            size_t k,
            const Label &label
    ) const
    {
        return _statistics.at( k ).accuracy( label );
    }

    double precision(
            size_t k,
            const Label &label
    ) const
    {
        return _statistics.at( k ).precision( label );
    }

    double recall(
            size_t k,
            const Label &label
    ) const
    {
        return _statistics.at( k ).recall( label );
    }

    double specificity(
            size_t k,
            const Label &label
    ) const
    {
        return _statistics.at( k ).specificity( label );
    }

    double fScore(
            size_t k,
            const Label &label,
            double beta = 1
    ) const
    {
        return _statistics.at( k ).fScore( label );
    }

    double mcc(
            size_t k,
            const Label &label
    ) const
    {
        return _statistics.at( k ).mcc( label );
    }

    std::pair<double, double> accuracy( const Label &label ) const
    {
        return averagingFunction( [&, this]( size_t k ) {
            return accuracy( k, label );
        } );
    }

    std::pair<double, double> precision( const Label &label ) const
    {
        return averagingFunction( [&, this]( size_t k ) {
            return precision( k, label );
        } );
    }

    std::pair<double, double> recall( const Label &label ) const
    {
        return averagingFunction( [&, this]( size_t k ) {
            return recall( k, label );
        } );
    }

    std::pair<double, double> specificity( const Label &label ) const
    {
        return averagingFunction( [&, this]( size_t k ) {
          return specificity( k, label );
        } );
    }

    std::pair<double, double> fScore(
            const Label &label,
            double beta = 1
    ) const
    {
        return averagingFunction( [&, this]( size_t k ) {
            return fScore( k, label, beta );
        } );
    }

    std::pair<double, double> mcc( const Label &label ) const
    {
        return averagingFunction( [&, this]( size_t k ) {
            return mcc( k, label );
        } );
    }

    std::pair<double, double> averageAccuracy() const
    {
        return averagingFunction( [this]( size_t k ) {
            return averageAccuracy( k );
        } );
    }

    std::pair<double, double> overallAccuracy() const
    {
        return averagingFunction( [this]( size_t k ) {
            return overallAccuracy( k );
        } );
    }

    std::pair<double, double> microPrecision() const
    {
        return averagingFunction( [this]( size_t k ) {
            return microPrecision( k );
        } );
    }

    std::pair<double, double> microRecall() const
    {
        return averagingFunction( [this]( size_t k ) {
            return microRecall( k );
        } );
    }

    std::pair<double, double> microSpecificity() const
    {
        return averagingFunction( [this]( size_t k ) {
          return microSpecificity( k );
        } );
    }

    std::pair<double, double> microFScore( double beta = 1 ) const
    {
        return averagingFunction( [this]( size_t k ) {
            return microFScore( k );
        } );
    }

    std::pair<double, double> macroPrecision() const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroPrecision( k );
        } );
    }

    std::pair<double, double> macroRecall() const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroRecall( k );
        } );
    }

    std::pair<double, double> macroSpecificity() const
    {
        return averagingFunction( [this]( size_t k ) {
          return macroSpecificity( k );
        } );
    }

    std::pair<double, double> macroFScore( double beta = 1 ) const
    {
        return averagingFunction( [this]( size_t k ) {
            return macroFScore( k );
        } );
    }

    std::pair<double, double> mcc() const
    {
        return averagingFunction( [this]( size_t k ) {
            return mcc( k );
        } );
    }

    std::map<Label, size_t> getLabelsWithCounts() const
    {
        std::map<Label, size_t> labelsWCounts;
        for (auto &fold : _statistics)
            for (auto &[l, count] : fold.getLabelsWithCounts())
                labelsWCounts[l] += count;

        return labelsWCounts;
    }

    std::map<Label, std::vector<std::pair<Label, size_t >>>
    misclassifications() const
    {
        std::map<Label, std::map<Label, size_t>> misclassifiedAcc;
        for (size_t i = 0; i < _k; ++i)
        {
            for (auto &[label, missed] : misclassifications( i ))
            {
                for (auto &[missedLabel, hits] : missed)
                    misclassifiedAcc[label][missedLabel] += hits;
            }
        }

        std::map<Label, std::vector<std::pair<Label, size_t >>> misclassified;
        for (auto&[label, missed] : misclassifiedAcc)
        {
            auto &_misclassified = misclassified[label];
            for (auto &[missedLabel, hits] : missed)
                _misclassified.emplace_back( missedLabel, hits );
            std::sort( _misclassified.begin(), _misclassified.end(),
                       [](
                               auto &p1,
                               auto &p2
                       ) { return p1.second > p2.second; } );
        }

        return misclassified;
    }


    void printReport(
            size_t indentation = 0,
            std::string_view tag = std::string_view()) const
    {
        fmt::print( "{:<{}}General Statistics [{}]:\n", "", indentation, tag );

        auto printRow = _printRowFunction( indentation + 2 );

        printRow( "Overall Accuracy", overallAccuracy());
        printRow( "Average Accuracy", averageAccuracy());
        printRow( "Macro Precision, Positive Predictive Value (PPV)", macroPrecision());
        printRow( "Micro Precision, Positive Predictive Value (PPV)", microPrecision());
        printRow( "Macro TPR, Recall, Sensitivity", macroRecall());
        printRow( "Micro TPR, Recall, Sensitivity", microRecall());
        printRow( "Macro TNR, Specificity", macroSpecificity());
        printRow( "Micro TNR, Specificity", microSpecificity());
        printRow( "Macro F1-Score", macroFScore());
        printRow( "Micro F1-Score", microFScore());
        printRow( "MCC (multiclass)", mcc());

        auto labels = getLabelsWithCounts();

        fmt::print( "{:<{}}Misclassification:\n", "", indentation + 2 );
        auto printRow2 = _printRowFunction( indentation + 4 );
        for (auto &[label, missclassified] : misclassifications())
        {
            std::vector<std::string> pairs;
            for (auto &[missedLabel, hits] : missclassified)
                pairs.push_back( fmt::format( "[{}:{}]", missedLabel, hits ));

            auto row = fmt::format( "Class ({}):{} -> {}", labels[label], label, io::join( pairs, "" ));

            fmt::print( "{:<{}}{}\n", "", indentation + 4, row );
        }
    }

    void printPerClassReport(
            size_t indentation = 0,
            std::string_view tag = std::string_view()) const
    {
        fmt::print( "{:<{}}Per-class Statistics [{}]:\n", "", indentation, tag );


        const auto labels = getLabelsWithCounts();
        auto scoresVector = [&, this]( std::function<std::pair<double, double>( const Label & )> &&fn ) {
            std::vector<std::pair<double, double>> scores;
            for (auto &[l, count] : labels)
                scores.emplace_back( fn( l ));
            return scores;
        };

        std::vector<std::string> labelsWCounts;
        size_t maxWidth = 0;
        for (auto &[l, count] : labels)
        {
            labelsWCounts.push_back( fmt::format( "{}({})", l, count ));
            maxWidth = std::max( maxWidth, labelsWCounts.back().size());
        }

        auto printHeader = _printHeaderColumnsFunction<decltype( labelsWCounts )>( indentation + 2, maxWidth + 1 );
        printHeader( "Criteria", labelsWCounts );

        auto printRow = _printColumnsFunction( indentation + 2, maxWidth + 1 );
        printRow( "Accuracy", scoresVector( [this]( const auto &l ) { return accuracy( l ); } ));
        printRow( "Precision", scoresVector( [this]( const auto &l ) { return precision( l ); } ));
        printRow( "Recall", scoresVector( [this]( const auto &l ) { return recall( l ); } ));
        printRow( "Specificity", scoresVector( [this]( const auto &l ) { return specificity( l ); } ));
        printRow( "F1-Score", scoresVector( [this]( const auto &l ) { return fScore( l, 1.0 ); } ));
        printRow( "MCC", scoresVector( [this]( const auto &l ) { return mcc( l ); } ));
    }

    template<size_t indentation = 0>
    void printReport( size_t k ) const
    {
        _statistics.at( k ).template printReport<indentation>();
    }

private:
    static auto _printRowFunction(
            size_t indentation,
            size_t col1Width = 50
    )
    {
        return [=](
                const char *col1,
                std::pair<double, double> &&col2
        ) {
            constexpr const char *fmtSpec = "{:<{}}:{:.4f}  (±{:.3f})";
            fmt::print( "{:<{}}{}\n", "", indentation,
                        fmt::format( fmtSpec, col1, col1Width, col2.first, col2.second ));
        };
    }

    template<typename SequenceContainer>
    static auto _printHeaderColumnsFunction(
            size_t indentation,
            size_t colWidth
    )
    {
        return [=](
                const char *col1,
                const SequenceContainer &columns
        ) {
            constexpr const char *colfmt = "{:<{}}";
            fmt::print( "{:<{}}{:<{}}", "", indentation, col1, colWidth );
            for (auto &col : columns)
                fmt::print( colfmt, col, colWidth );
            fmt::print( "\n" );
        };
    }

    static auto _printColumnsFunction(
            size_t indentation,
            size_t colWidth = 20
    )
    {
        return [=](
                const char *col1,
                const std::vector<std::pair<double, double>> &columns
        ) {
            constexpr const char *colfmt = "{:<{}}";
            constexpr const char *cellfmt = "{:.4f}  (±{:.3f})";
            fmt::print( "{:<{}}{:<{}}", "", indentation, col1, colWidth );
            for (auto &col : columns)
                fmt::print( colfmt, fmt::format( cellfmt, col.first, col.second ), colWidth );
            fmt::print( "\n" );
        };
    }

private:
    size_t _k;
    std::vector<ConfusionMatrix<Label>> _statistics;
};


#endif //MARKOVIAN_FEATURES_CROSSVALIDATIONSTATISTICS_HPP
