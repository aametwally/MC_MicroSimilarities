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
public:
    ConfusionMatrix(size_t order) :
            _order(order),
            _matrix(Matrix(_order, Row(_order, 0)))
    {
    }

    ConfusionMatrix(const std::vector<Label> &labels) :
            _order(labels.size()),
            _matrix(Matrix(_order, Row(_order, 0)))
    {
        defineLabels(labels);
    }

    void defineLabels(const std::vector<Label> &labels)
    {
        assert(_dictionary.empty());
        assert(labels.size() == _order);

        size_t i = 0;
        for (auto &label : labels)
            _dictionary[label] = i++;
    }

    void countInstance(const Label &output, const Label &actual)
    {
        assert(!_dictionary.empty());
        size_t actualIdx = _dictionary.at(actual);
        size_t outputIdx = _dictionary.at(output);
        ++_matrix[actualIdx][outputIdx];
    }


    size_t truePositives(const Label &cl) const
    {
        assert(!_dictionary.empty());
        size_t classIdx = _dictionary.at(cl);
        return _matrix.at(classIdx).at(classIdx);
    }

    size_t falsePositives(const Label &cl) const
    {
        size_t classIdx = _dictionary.at(cl);
        size_t fp = 0;
        for (auto i = 0; i < _order; ++i)
            fp += _matrix.at(i).at(classIdx);
        return fp - truePositives(cl);
    }

    size_t falseNegatives(const Label &cl) const
    {
        size_t classIdx = _dictionary.at(cl);
        size_t fn = 0;
        for (auto i = 0; i < _order; ++i)
            fn += _matrix.at(classIdx).at(i);
        return fn - truePositives(cl);
    }

    size_t trueNegatives(const Label &cl) const
    {
        auto classIdx = _dictionary.at(cl);
        return _trueNegatives(classIdx);
    }

    size_t population(const Label &cl) const
    {
        size_t classIdx = _dictionary.at(cl);
        return _population(classIdx);
    }

    size_t population() const
    {
        return std::accumulate(_matrix.cbegin(), _matrix.cend(),
                               size_t(0), [](size_t count, const Row &row) {
                    return count + _rowCount(row);
                });
    }

    double accuracy(const Label &cl) const
    {
        size_t classIdx = _dictionary.at(cl);
        return _accuracy(classIdx);
    }

    double averageAccuracy() const
    {
        double acc = 0;
        for (auto i = 0; i < _order; ++i)
            acc += _accuracy(i);
        return acc / _order;
    }

    double precision(const Label &cl) const
    {
        auto classIdx = _dictionary.at(cl);
        return _precision(classIdx);
    }

    double microPrecision() const
    {
        size_t allTp = _allTruePositives(), allFp = _allFalsePositives();
        return double(allTp) / (allTp + allFp);
    }

    double recall(const Label &cl)
    {
        auto classIdx = _dictionary.at(cl);
        return _recall(classIdx);
    }

    double microRecall() const
    {
        size_t allTp = _allTruePositives(), allFn = _allFalseNegatives();
        return double(allTp) / (allTp + allFn);
    }

    double fScore(const Label &cl, double beta = 1) const
    {
        auto classIdx = _dictionary.at(cl);
        return _fScore(classIdx, beta);
    }

    double microFScore(double beta = 1) const
    {
        auto mPrecision = microPrecision(), mRecall = microRecall();
        beta *= beta;
        return (beta + 1) * mPrecision * mRecall / (beta * mPrecision + mRecall);
    }

    double macroPrecision() const
    {
        double mPrecision = 0;
        for (auto i = 0; i < _order; ++i)
            mPrecision += _precision(i);
        return mPrecision / _order;
    }

    double macroRecall() const
    {
        double mRecall = 0;
        for (auto i = 0; i < _order; ++i)
            mRecall += _recall(i);
        return mRecall / _order;
    }

    double macroFScore(double beta = 1) const
    {
        auto mPrecision = macroPrecision(), mRecall = macroRecall();
        beta *= beta;
        return (beta + 1) * mPrecision * mRecall / (beta * mPrecision + mRecall);
    }

    double accuracy() const
    {
        return double(_allTrue()) / population();
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
        return double(allTp) / (allTp + allFn);
    }

    void printReport(std::string indentation = "") const
    {
        fmt::print("{1}General Statistics:\n", indentation);
        indentation += "  ";
        fmt::print("{1}Overall Accuracy:\t{2}\n"
                   "{1}Macro Precision, Positive Predictive Value (PPV):\t{3}\n"
                   "{1}Micro Precision, Positive Predictive Value (PPV):\t{4}\n"
                   "{1}Macro TPR, Recall, Sensitivity:\t{5}\n"
                   "{1}Micro TPR, Recall, Sensitivity:\t{6}\n"
                   "{1}Macro F1-Score:\t{7}\n"
                   "{1}Micro F1-Score:\t{8}\n",
                   indentation,
                   averageAccuracy(),
                   macroPrecision(), microPrecision(),
                   macroRecall(), microRecall(),
                   macroFScore(), microFScore());
    }

    void printReport(const Label &cl, std::string indentation = "") const
    {
        auto classIdx = _dictionary.at(cl);
        fmt::print("{1}General Statistics for class[{2}]:[{3}]",
                   indentation, classIdx, cl);
        indentation += "  ";
        fmt::print("{1}Accuracy:\t{1}\n"
                   "{1}F0.5 Score:\t{2}\n"
                   "{1}F1 Score:\t{3}\n"
                   "{1}F2 Score:\t{4}\n"
                   "{1}Population:\t{5}\n"
                   "{1}Precision, PPV:\t{6}\n"
                   "{1}Recall, sensitivity, TPR:\t{7}\n",
                   indentation,
                   _accuracy(classIdx),
                   _fScore(classIdx, 0.5), _fScore(classIdx, 1), _fScore(classIdx, 2),
                   _population(classIdx),
                   _precision(classIdx),
                   _recall(classIdx));
    }

private:
    double _accuracy(size_t classIdx) const
    {
        auto tp = _truePositives(classIdx);
        auto tn = _trueNegatives(classIdx);
        auto fp = _falsePositives(classIdx);
        auto fn = _falseNegatives(classIdx);
        return double(tp + tn) / (tp + tn + fp + fn);
    }

    size_t _truePositives(size_t classIdx) const
    {
        return _matrix.at(classIdx).at(classIdx);
    }

    size_t _trueNegatives(size_t classIdx) const
    {
        return _allTrue() - _truePositives(classIdx);
    }

    size_t _falsePositives(size_t classIdx) const
    {
        size_t fp = 0;
        for (auto i = 0; i < _order; ++i)
            fp += _matrix.at(i).at(classIdx);
        return fp - _truePositives(classIdx);
    }

    size_t _falseNegatives(size_t classIdx) const
    {
        const auto &row = _matrix.at(classIdx);
        size_t fn = std::accumulate(row.cbegin(), row.cend(), size_t(0));
        return fn - _truePositives(classIdx);
    }

    size_t _population(size_t classIdx) const
    {
        const auto &row = _matrix.at(classIdx);
        return _rowCount(row);
    }

    size_t _rowCount(const Row &row) const
    {
        return std::accumulate(row.cbegin(), row.cend(), size_t(0));
    }

    size_t _allTrue() const
    {
        size_t allTrue = 0;
        for (auto i = 0; i < _order; ++i)
            allTrue += _matrix.at(_order).at(_order);
        return allTrue;
    }

    size_t _allTruePositives() const
    {
        size_t allTp = 0;
        for (auto i = 0; i < _order; ++i)
            allTp += _truePositives(i);
        return allTp;
    }

    size_t _allFalseNegatives() const
    {
        size_t allFn = 0;
        for (auto i = 0; i < _order; ++i)
            allFn += _falseNegatives(i);
        return allFn;
    }

    size_t _allFalsePositives() const
    {
        size_t allFp = 0;
        for (auto i = 0; i < _order; ++i)
            allFp += _falsePositives(i);
        return allFp;
    }

    double _precision(size_t classIdx) const
    {
        auto tp = _truePositives(classIdx), fp = _falsePositives(classIdx);
        return double(tp) / (tp + fp);
    }

    double _recall(size_t classIdx) const
    {
        auto tp = _truePositives(classIdx), fn = _falseNegatives(classIdx);
        return double(tp) / (tp + fn);
    }

    double _fScore(size_t classIdx, double beta) const
    {
        auto mPrecision = _precision(classIdx), mRecall = _recall(classIdx);
        beta *= beta;
        return (beta + 1) * mPrecision * mRecall / (beta * mPrecision + mRecall);
    }

private:
    const size_t _order;
    Matrix _matrix;
    std::map<Label, size_t> _dictionary;
};


#endif //MARKOVIAN_FEATURES_CONFUSIONMATRIX_HPP
