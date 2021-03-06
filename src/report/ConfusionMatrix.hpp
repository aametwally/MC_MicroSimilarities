//
// Created by asem on 01/08/18.
//

#ifndef MARKOVIAN_FEATURES_CONFUSIONMATRIX_HPP
#define MARKOVIAN_FEATURES_CONFUSIONMATRIX_HPP

#include "common.hpp"

template < typename Label = std::string, typename T = int64_t >
class ConfusionMatrix
{
private:
    using Row = std::vector<T>;
    using Matrix = std::vector<Row>;
    static constexpr double eps = std::numeric_limits<double>::epsilon();

public:
    explicit ConfusionMatrix( const std::set<Label> &labels ) :
            _dictionary( _makeDictionary( labels )),
            _order( _dictionary.size()),
            _matrix( Matrix( _order, Row( _order, 0 ))) {}

    template < typename U >
    static ConfusionMatrix<std::string, U>
    fromRawConfusionMatrix( const std::vector<std::vector<U>> &raw )
    {
        assert( !raw.empty());
        auto nRows = raw.size();
        auto nCols = raw.front().size();
        assert( nRows == nCols && std::all_of( raw.cbegin(), raw.cend(),
                                               [=]( auto &&row )
                                               {
                                                 return row.size() == nCols;
                                               } ));
        std::set<std::string> labels;

        for ( auto i = 0; i < nRows; ++i )
            labels.insert( fmt::format( "Label#{}", i ));

        auto cm = ConfusionMatrix<std::string, U>( labels );

        if ( labels.size() == cm._order - 1 )
        {
            auto skipIndex = cm._dictionary.at( cm._unclassifiedLabel());
            for ( auto iLocal = 0, iExternal = 0;
                  iLocal < cm._order && iExternal < labels.size();
                  ++iLocal, ++iExternal )
            {
                if ( iLocal == skipIndex ) ++iLocal;
                for ( auto jLocal = 0, jExternal = 0;
                      jLocal < cm._order && jExternal < labels.size();
                      ++jLocal, ++jExternal )
                {
                    if ( jLocal == skipIndex ) ++jLocal;

                    cm._matrix[iLocal][jLocal] = raw.at( iExternal ).at( jExternal );
                }
            }
        } else if ( labels.size() == cm._order )
        {
            cm._matrix = raw;
        } else throw std::runtime_error( "Unexpected case!" );

        return cm;
    }

    void countInstance(
            const Label &prediction,
            const Label &actual
    )
    {
        assert( !_dictionary.empty());
        size_t actualIdx = _getClassIdx( actual );
        size_t outputIdx = _getClassIdx( prediction );
        ++_matrix[actualIdx][outputIdx];
    }

    T truePositives( const Label &cl ) const
    {
        assert( !_dictionary.empty());
        size_t classIdx = _getClassIdx( cl );
        return _matrix.at( classIdx ).at( classIdx );
    }

    T falsePositives( const Label &cl ) const
    {
        size_t classIdx = _getClassIdx( cl );
        T fp = 0;
        for ( auto i = 0; i < _order; ++i )
            fp += _matrix.at( i ).at( classIdx );
        return fp - truePositives( cl );
    }

    T falseNegatives( const Label &cl ) const
    {
        size_t classIdx = _getClassIdx( cl );
        T fn = 0;
        for ( auto i = 0; i < _order; ++i )
            fn += _matrix.at( classIdx ).at( i );
        return fn - truePositives( cl );
    }

    T trueNegatives( const Label &cl ) const
    {
        auto classIdx = _getClassIdx( cl );
        return _trueNegatives( classIdx );
    }

    T population( const Label &cl ) const
    {
        size_t classIdx = _getClassIdx( cl );
        return _population( classIdx );
    }

    T population() const
    {
        return std::accumulate( _matrix.cbegin(), _matrix.cend(),
                                T( 0 ), [this](
                        T count,
                        const Row &row
                )
                                {
                                  return count + _rowCount( row );
                                } );
    }

    double accuracy( const Label &cl ) const
    {
        size_t classIdx = _getClassIdx( cl );
        return _accuracy( classIdx );
    }

    double auc( const Label &cl ) const
    {
        size_t classIdx = _getClassIdx( cl );
        return _auc( classIdx );
    }

    double mcc( const Label &cl ) const
    {
        return _mcc( _dictionary.at( cl ));
    }

    double averageAccuracy() const
    {
        double acc = 0;
        for ( auto i = 0; i < _order; ++i )
            acc += _accuracy( i );
        return acc / _order;
    }

    double overallAccuracy() const
    {
        double acc = 0;
        for ( auto i = 0; i < _order; ++i )
            acc += _truePositives( i );
        return acc / population();
    }

    double precision( const Label &cl ) const
    {
        auto classIdx = _getClassIdx( cl );
        return _precision( classIdx );
    }

    double microPrecision() const
    {
        T allTp = _allTruePositives(), allFp = _allFalsePositives();
        return double( allTp + eps ) / ( allTp + allFp + eps );
    }

    double microSpecificity() const
    {
        T allTn = _allTrueNegatives(), allFp = _allFalsePositives();
        return double( allTn + eps ) / ( allTn + allFp + eps );
    }

    double specificity( const Label &cl ) const
    {
        auto classIdx = _getClassIdx( cl );
        return _specificity( classIdx );
    }

    double recall( const Label &cl ) const
    {
        auto classIdx = _getClassIdx( cl );
        return _recall( classIdx );
    }

    double microRecall() const
    {
        T allTp = _allTruePositives(), allFn = _allFalseNegatives();
        return double( allTp + eps ) / ( allTp + allFn + eps );
    }

    double fScore(
            const Label &cl,
            double beta = 1
    ) const
    {
        auto classIdx = _getClassIdx( cl );
        return _fScore( classIdx, beta );
    }

    double microFScore( double beta = 1 ) const
    {
        auto mPrecision = microPrecision(), mRecall = microRecall();
        beta *= beta;
        return (( beta + 1 ) * mPrecision * mRecall + eps ) / ( beta * mPrecision + mRecall + eps );
    }

    double macroPrecision() const
    {
        double mPrecision = 0;
        for ( auto i = 0; i < _order; ++i )
            mPrecision += _precision( i );
        return mPrecision / _order;
    }

    double macroRecall() const
    {
        double mRecall = 0;
        for ( auto i = 0; i < _order; ++i )
            mRecall += _recall( i );
        return mRecall / _order;
    }

    double macroSpecificity() const
    {
        double mRecall = 0;
        for ( auto i = 0; i < _order; ++i )
            mRecall += _specificity( i );
        return mRecall / _order;
    }

    double macroFScore( double beta = 1 ) const
    {
        auto mPrecision = macroPrecision(), mRecall = macroRecall();
        beta *= beta;
        return (( beta + 1 ) * mPrecision * mRecall + eps ) / ( beta * mPrecision + mRecall + eps );
    }

    /**
     * https://en.wikipedia.org/wiki/Matthews_correlation_coefficient#Multiclass_case
     * @return
     */
    double mcc() const
    {
        double num = 0;
        const auto &c = _matrix;
        for ( auto k = 0; k < _order; ++k )
            for ( auto l = 0; l < _order; ++l )
                for ( auto m = 0; m < _order; ++m )
                    num += ( c[k][k] * c[l][m] - c[k][l] * c[m][k] );

        double den1 = 0;
        for ( auto k = 0; k < _order; ++k )
        {
            double den1a = 0;
            for ( auto l = 0; l < _order; ++l )
                den1a += c[k][l];
            double den1b = 0;
            for ( auto kk = 0; kk < _order; ++kk )
            {
                if ( kk == k ) continue;
                else
                {
                    for ( auto ll = 0; ll < _order; ++ll )
                        den1b += c[kk][ll];
                }
            }
            den1 += ( den1a * den1b );
        }
        double den2 = 0;
        for ( auto k = 0; k < _order; ++k )
        {
            double den2a = 0;
            for ( auto l = 0; l < _order; ++l )
                den2a += c[l][k];
            double den2b = 0;
            for ( auto kk = 0; kk < _order; ++kk )
            {
                if ( kk == k ) continue;
                else
                {
                    for ( auto ll = 0; ll < _order; ++ll )
                        den2b += c[ll][kk];
                }
            }
            den2 += ( den2a * den2b );
        }

        return num / std::sqrt( den1 * den2 + eps );
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
        return double( allTp + eps ) / ( allTp + allFn + eps );
    }

    std::map<Label, std::map<Label, size_t >> misclassifications() const
    {
        std::map<Label, std::map<Label, size_t >> missclassified;
        for ( auto &[label, idx] : _dictionary )
        {
            for ( auto &[misslabel, hits] : _misclassifications( idx ))
                missclassified[label][misslabel] += hits;
        }
        return missclassified;
    }

    std::map<Label, size_t> misclassifications( const Label &l ) const
    {
        auto classIdx = _getClassIdx( l );
        return _misclassifications( classIdx );
    }

    std::vector<Label> getLabels() const
    {
        std::vector<Label> ls;
        for ( auto &[label, i] : _dictionary )
            if ( label != _unclassifiedLabel())
                ls.push_back( label );
        return ls;
    }

    std::map<Label, T> getLabelsWithCounts() const
    {
        std::map<Label, T> ls;
        for ( auto &[label, index] : _dictionary )
            if ( label != _unclassifiedLabel())
                ls.emplace( label, _population( index ));
        return ls;
    }

    template < size_t indentation = 0 >
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
        printRow( "Macro TNR, Specificity", macroSpecificity());
        printRow( "Micro TNR, Specificity", microSpecificity());
        printRow( "Macro F1-Score", macroFScore());
        printRow( "Micro F1-Score", microFScore());
        printRow( "MCC (multiclass)", mcc());
    }

    template < size_t indentation = 0 >
    void printClassReport( const Label &cl ) const
    {
        auto classIdx = _getClassIdx( cl );
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
        printRow( "Specificity, TNR", _specificity( classIdx ));
        printRow( "AUC", _auc( classIdx ));
        printRow( "MCC", _mcc( classIdx ));
    }

private:
    Label _getClassLabel( size_t index ) const
    {
        for ( auto &[label, idx] : _dictionary )
            if ( index == idx ) return label;
        return {};
    }

    size_t _getClassIdx( const Label &cl ) const
    {
        try
        {
            auto idx = _dictionary.at( cl );
            return idx;
        } catch ( const std::out_of_range & )
        {
            return _dictionary.at( _unclassifiedLabel());
        }
    }

    template < size_t indentation, size_t col1Width = 50 >
    static auto _printRowFunction()
    {
        return [=](
                const char *col1,
                double col2
        )
        {
          constexpr const char *fmtSpec = "{:<{}}{:<{}}:{}\n";
          fmt::print( fmtSpec, "", indentation, col1, col1Width, col2 );
        };
    }

    static std::map<Label, size_t> _makeDictionary( const std::set<Label> &labels )
    {
        std::map<Label, size_t> dic;
        size_t i = 0;
        for ( auto &label : labels )
            dic.emplace( label, i++ );
        dic.emplace( _unclassifiedLabel(), i );

        return dic;
    }


    double _accuracy( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx );
        auto tn = _trueNegatives( classIdx );
        auto fp = _falsePositives( classIdx );
        auto fn = _falseNegatives( classIdx );
        return double( tp + tn ) / ( tp + tn + fp + fn + eps );
    }

    T _truePositives( size_t classIdx ) const
    {
        return _matrix.at( classIdx ).at( classIdx );
    }

    T _trueNegatives( size_t classIdx ) const
    {
        return _diagonal() - _truePositives( classIdx );
    }

    T _falsePositives( size_t classIdx ) const
    {
        T fp = 0;
        for ( auto i = 0; i < _order; ++i )
            fp += _matrix.at( i ).at( classIdx );
        return fp - _truePositives( classIdx );
    }

    T _falseNegatives( size_t classIdx ) const
    {
        const auto &row = _matrix.at( classIdx );
        T fn = std::accumulate( row.cbegin(), row.cend(), T( 0 ));
        return fn - _truePositives( classIdx );
    }

    T _population( size_t classIdx ) const
    {
        const auto &row = _matrix.at( classIdx );
        return _rowCount( row );
    }

    T _rowCount( const Row &row ) const
    {
        return std::accumulate( row.cbegin(), row.cend(), T( 0 ));
    }

    T _diagonal() const
    {
        T allTrue = 0;
        for ( auto i = 0; i < _order; ++i )
            allTrue += _matrix.at( i ).at( i );
        return allTrue;
    }

    T _allTruePositives() const
    {
        T allTp = 0;
        for ( auto i = 0; i < _order; ++i )
            allTp += _truePositives( i );
        return allTp;
    }

    T _allTrueNegatives() const
    {
        T allTn = 0;
        for ( auto i = 0; i < _order; ++i )
            allTn += _trueNegatives( i );
        return allTn;
    }

    T _allFalseNegatives() const
    {
        T allFn = 0;
        for ( auto i = 0; i < _order; ++i )
            allFn += _falseNegatives( i );
        return allFn;
    }

    T _allFalsePositives() const
    {
        T allFp = 0;
        for ( auto i = 0; i < _order; ++i )
            allFp += _falsePositives( i );
        return allFp;
    }

    double _mcc( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx );
        auto tn = _trueNegatives( classIdx );
        auto fp = _falsePositives( classIdx );
        auto fn = _falseNegatives( classIdx );
        auto den = ( tp + fp ) * ( tp + fn ) * ( tn + fp ) * ( tn + fn );
        if ( den == 0 ) den = 1;
        return ( tp * tn - fp * fn ) / std::sqrt( den );
    }

    double _precision( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx ), fp = _falsePositives( classIdx );
        return double( tp + eps ) / ( tp + fp + eps );
    }

    double _recall( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx ), fn = _falseNegatives( classIdx );
        return double( tp + eps ) / ( tp + fn + eps );
    }

    double _specificity( size_t classIdx ) const
    {
        auto tn = _trueNegatives( classIdx ), fp = _falsePositives( classIdx );
        return double( tn + eps ) / ( tn + fp + eps );
    }


    double _fScore(
            size_t classIdx,
            double beta
    ) const
    {
        auto mPrecision = _precision( classIdx ), mRecall = _recall( classIdx );
        beta *= beta;
        return (( beta + 1 ) * mPrecision * mRecall + eps ) / ( beta * mPrecision + mRecall + eps );
    }

    double _auc( size_t classIdx ) const
    {
        auto tp = _truePositives( classIdx );
        auto tn = _trueNegatives( classIdx );
        auto fp = _falsePositives( classIdx );
        auto fn = _falseNegatives( classIdx );

        return ( double( tp + eps ) / ( tp + fn + eps ) + double( tn + eps ) / ( tn + fp + eps )) / 2;
    }

    std::map<Label, size_t> _misclassifications( size_t classIdx ) const
    {
        std::map<Label, size_t> misclassified;
        auto &row = _matrix.at( classIdx );
        for ( auto &[label, col] : _dictionary )
        {
            if ( col == classIdx ) continue;
            else misclassified[label] = row.at( col );
        }
        return misclassified;
    }

    template < typename L = Label, typename std::enable_if<!std::is_arithmetic<L>::value, void>::type * = nullptr >
    static Label _unclassifiedLabel()
    {
        static Label unclassified = Label();
        return unclassified;
    }

    template < typename L = Label, typename std::enable_if<std::is_arithmetic<L>::value, void>::type * = nullptr >
    static Label _unclassifiedLabel()
    {
        return std::numeric_limits<L>::quiet_NaN();
    }

private:
    const std::map<Label, size_t> _dictionary;
    const size_t _order;
    Matrix _matrix;
};


#endif //MARKOVIAN_FEATURES_CONFUSIONMATRIX_HPP
