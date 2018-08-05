//
// Created by asem on 01/08/18.
//

#ifndef MARKOVIAN_FEATURES_CONFUSIONMATRIX_HPP
#define MARKOVIAN_FEATURES_CONFUSIONMATRIX_HPP

#include "common.hpp"

template<typename Label = std::string>
class ConfusionMatrix
{
private:
    using Row = std::vector<size_t>;
    using Matrix = std::vector<Row>;
    static constexpr double eps = std::numeric_limits<double>::epsilon();
public:

    explicit ConfusionMatrix( const std::set<Label> &labels ) :
            _dictionary( _makeDictionary( labels )),
            _order( _dictionary.size()),
            _matrix( Matrix( _order, Row( _order, 0 )))
    {

    }

    explicit ConfusionMatrix( const std::vector<Label> &labels, const std::vector<Label> &prediction )
            : _dictionary( _makeDictionary( labels, prediction )),
              _order( _dictionary.size()),
              _matrix( Matrix( _order, Row( _order, 0 )))
    {
        assert( labels.size() == prediction.size());

        if ( labels.size() != prediction.size())
            throw std::length_error( "Labels size should match predictions size" );
        else
        {
            for (auto i = 0; i < labels.size(); ++i)
                countInstance( prediction.at( i ), labels.at( i ));
        }
    }

    void countInstance( const Label &prediction, const Label &actual )
    {
        assert( !_dictionary.empty());
        size_t actualIdx = _dictionary.at( actual );
        size_t outputIdx = _dictionary.at( prediction );
        ++_matrix[actualIdx][outputIdx];
    }

    size_t truePositives( const Label &cl ) const
    {
        assert( !_dictionary.empty());
        size_t classIdx = _dictionary.at( cl );
        return _matrix.at( classIdx ).at( classIdx );
    }

    size_t falsePositives( const Label &cl ) const
    {
        size_t classIdx = _dictionary.at( cl );
        size_t fp = 0;
        for (auto i = 0; i < _order; ++i)
            fp += _matrix.at( i ).at( classIdx );
        return fp - truePositives( cl );
    }

    size_t falseNegatives( const Label &cl ) const
    {
        size_t classIdx = _dictionary.at( cl );
        size_t fn = 0;
        for (auto i = 0; i < _order; ++i)
            fn += _matrix.at( classIdx ).at( i );
        return fn - truePositives( cl );
    }

    size_t trueNegatives( const Label &cl ) const
    {
        auto classIdx = _dictionary.at( cl );
        return _trueNegatives( classIdx );
    }

    size_t population( const Label &cl ) const
    {
        size_t classIdx = _dictionary.at( cl );
        return _population( classIdx );
    }

    size_t population() const
    {
        return std::accumulate( _matrix.cbegin(), _matrix.cend(),
                                size_t( 0 ), [this]( size_t count, const Row &row ) {
                    return count + _rowCount( row );
                } );
    }

    double accuracy( const Label &cl ) const
    {
        size_t classIdx = _dictionary.at( cl );
        return _accuracy( classIdx );
    }

    double auc( const Label &cl ) const
    {
        size_t classIdx = _dictionary.at( cl );
        return _auc( classIdx );
    }

    double averageAccuracy() const
    {
        double acc = 0;
        for (auto i = 0; i < _order; ++i)
            acc += _accuracy( i );
        return acc / _order;
    }

    double overallAccuracy() const
    {
        double acc = 0;
        for (auto i = 0; i < _order; ++i)
            acc += _truePositives( i );
        return acc / population();
    }

    double precision( const Label &cl ) const
    {
        auto classIdx = _dictionary.at( cl );
        return _precision( classIdx );
    }

    double microPrecision() const
    {
        size_t allTp = _allTruePositives(), allFp = _allFalsePositives();
        return double( allTp + eps ) / (allTp + allFp + eps);
    }

    double recall( const Label &cl )
    {
        auto classIdx = _dictionary.at( cl );
        return _recall( classIdx );
    }

    double microRecall() const
    {
        size_t allTp = _allTruePositives(), allFn = _allFalseNegatives();
        return double( allTp + eps ) / (allTp + allFn + eps);
    }

    double fScore( const Label &cl, double beta = 1 ) const
    {
        auto classIdx = _dictionary.at( cl );
        return _fScore( classIdx, beta );
    }

    double microFScore( double beta = 1 ) const
    {
        auto mPrecision = microPrecision(), mRecall = microRecall();
        beta *= beta;
        return ((beta + 1) * mPrecision * mRecall + eps) / (beta * mPrecision + mRecall + eps);
    }

    double macroPrecision() const
    {
        double mPrecision = 0;
        for (auto i = 0; i < _order; ++i)
            mPrecision += _precision( i );
        return mPrecision / _order;
    }

    double macroRecall() const
    {
        double mRecall = 0;
        for (auto i = 0; i < _order; ++i)
            mRecall += _recall( i );
        return mRecall / _order;
    }

    double macroFScore( double beta = 1 ) const
    {
        auto mPrecision = macroPrecision(), mRecall = macroRecall();
        beta *= beta;
        return ((beta + 1) * mPrecision * mRecall + eps) / (beta * mPrecision + mRecall + eps);
    }

    double accuracy() const
    {
        return double( _allTrue()) / population();
    }

    /**
     * @brief sensitivity
     * Aka recall, true positive rate (tpr)
     * @return
     */
    double sensitivity() const
    {
        auto allTp = _allTruePositives();
        auto allFn = _allFalseNegatives();
        return double( allTp + eps ) / (allTp + allFn + eps);
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

    template<size_t indentation = 0>
    void printClassReport( const Label &cl ) const
    {
        auto classIdx = _dictionary.at( cl );
        fmt::print( "{:<{}}General Statistics for class[{}]:[{}]\n",
                    "", indentation, classIdx, cl );
        auto printRow = _printRowFunction<indentation + 2>();

        printRow( "Accuracy", _accuracy( classIdx ));
        printRow( "F0.5 Score", _fScore( classIdx, 0.5 ));
        printRow( "F1 Score", _fScore( classIdx, 1 ));
        printRow( "F2 Score", _fScore( classIdx, 2 ));
        printRow( "Population", _population( classIdx ));
        printRow( "Precision, PPV", _precision( classIdx ));
        printRow( "Recall, sensitivity, TPR", _recall( classIdx ));
        printRow( "AUC", _auc( classIdx ));

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

    static std::map<Label, size_t> _makeDictionary( const std::set<Label> &labels )
    {
        std::map<Label, size_t> dic;
        size_t i = 0;
        for (auto &label : labels)
            dic.emplace( label, i++ );

        return dic;
    }

    static std::map<Label, size_t> _makeDictionary( const std::vector<Label> &labels,
                                                    const std::vector<Label> &prediction )
    {
        std::map<Label, size_t> dic;
        size_t i = 0;
        for (auto &label : labels)
        {
            auto res = dic.emplace( label, i );
            i += res.second;
        }
        for (auto &p : prediction)
        {
            auto res = dic.emplace( p, i );
            i += res.second;
        }
        return dic;
    }

    double _accuracy( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx );
        auto tn = _trueNegatives( classIdx );
        auto fp = _falsePositives( classIdx );
        auto fn = _falseNegatives( classIdx );
        return double( tp + tn ) / (tp + tn + fp + fn + eps);
    }

    size_t _truePositives( size_t classIdx ) const
    {
        return _matrix.at( classIdx ).at( classIdx );
    }

    size_t _trueNegatives( size_t classIdx ) const
    {
        return _allTrue() - _truePositives( classIdx );
    }

    size_t _falsePositives( size_t classIdx ) const
    {
        size_t fp = 0;
        for (auto i = 0; i < _order; ++i)
            fp += _matrix.at( i ).at( classIdx );
        return fp - _truePositives( classIdx );
    }

    size_t _falseNegatives( size_t classIdx ) const
    {
        const auto &row = _matrix.at( classIdx );
        size_t fn = std::accumulate( row.cbegin(), row.cend(), size_t( 0 ));
        return fn - _truePositives( classIdx );
    }

    size_t _population( size_t classIdx ) const
    {
        const auto &row = _matrix.at( classIdx );
        return _rowCount( row );
    }

    size_t _rowCount( const Row &row ) const
    {
        return std::accumulate( row.cbegin(), row.cend(), size_t( 0 ));
    }

    size_t _allTrue() const
    {
        size_t allTrue = 0;
        for (auto i = 0; i < _order; ++i)
            allTrue += _matrix.at( i ).at( i );
        return allTrue;
    }

    size_t _allTruePositives() const
    {
        size_t allTp = 0;
        for (auto i = 0; i < _order; ++i)
            allTp += _truePositives( i );
        return allTp;
    }

    size_t _allFalseNegatives() const
    {
        size_t allFn = 0;
        for (auto i = 0; i < _order; ++i)
            allFn += _falseNegatives( i );
        return allFn;
    }

    size_t _allFalsePositives() const
    {
        size_t allFp = 0;
        for (auto i = 0; i < _order; ++i)
            allFp += _falsePositives( i );
        return allFp;
    }

    double _precision( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx ), fp = _falsePositives( classIdx );
        return double( tp + eps ) / (tp + fp + eps);
    }

    double _recall( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx ), fn = _falseNegatives( classIdx );
        return double( tp + eps ) / (tp + fn + eps);
    }

    double _fScore( size_t classIdx, double beta ) const
    {
        auto mPrecision = _precision( classIdx ), mRecall = _recall( classIdx );
        beta *= beta;
        return ((beta + 1) * mPrecision * mRecall + eps) / (beta * mPrecision + mRecall + eps);
    }

    double _auc( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx );
        auto tn = _trueNegatives( classIdx );
        auto fp = _falsePositives( classIdx );
        auto fn = _falseNegatives( classIdx );

        return (double( tp + eps ) / (tp + fn + eps) + double( tn + eps ) / (tn + fp + eps)) / 2;
    }

private:
    const std::map<Label, size_t> _dictionary;
    const size_t _order;
    Matrix _matrix;
};


#endif //MARKOVIAN_FEATURES_CONFUSIONMATRIX_HPP
