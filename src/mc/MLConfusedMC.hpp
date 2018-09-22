//
// Created by asem on 13/09/18.
//

#ifndef MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
#define MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP

#include "common.hpp"
#include "AbstractMC.hpp"
#include "AbstractClassifier.hpp"

#include "dlib_utilities.hpp"
#include "dlib/svm.h"


namespace MC {
    class MLConfusedMC : public AbstractClassifier
    {

    public:
        MLConfusedMC() : _enableLDA( false )
        {}

        using FeatureVector = std::vector<double>;

        void enableLDA( std::optional<size_t> ldaDims = std::nullopt )
        {
            _enableLDA = true;
            _ldaDims = ldaDims;
        }

        void fit( const std::map<std::string_view, std::vector<std::string >> &training )
        {
            _normalizer = [this]( auto &&f ) { return _minmaxScaleColumns( std::move( f )); };

            std::vector<std::string_view> labels;
            std::vector<FeatureVector> featuresVector;
            for (auto &[trainLabel, trainSeqs] : training)
            {
                for (auto &seq : trainSeqs)
                {
                    auto features = _extractFeatures( seq );

                    assert( std::all_of( features.begin(), features.end(),
                                         []( double v ) { return !std::isnan( v ); } ));

                    featuresVector.emplace_back( features );
                    labels.push_back( trainLabel );
                }
            }
            if ( _normalizer )
            {
                _trainNormalizers( featuresVector );
                featuresVector = _normalizeFeatures( std::move( featuresVector ));
            }
            if ( _enableLDA )
            {
                _trainLDA( featuresVector, labels );
                featuresVector = _transformFeatures( std::move( featuresVector ));
            }
            _fitML( labels, std::move( featuresVector ));
        }

    protected:
        ScoredLabels _predict( std::string_view sequence ) const override
        {
            auto features = _extractFeatures( sequence );
            assert( std::all_of( features.begin(), features.end(),
                                 []( double v ) { return !std::isnan( v ); } ));

            if ( _normalizer )
                features = _normalizeFeatures( std::move( features ));

            if ( _enableLDA && _Z && _M )
                features = _transformFeatures( features );

            return _predictML( features );
        }

        virtual void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) = 0;

        virtual ScoredLabels _predictML( const FeatureVector &f ) const = 0;

        virtual FeatureVector _extractFeatures( std::string_view sequence ) const = 0;

    private:
        using NormalizerFunction = std::function<std::vector<double>( std::vector<double> && )>;

        static std::vector<size_t> _numericLabels( const std::vector<std::string_view> &labels )
        {
            std::set<std::string_view> uniqueLabels( labels.cbegin(), labels.cend());
            std::map<std::string_view, size_t> labelsMap;
            {
                size_t labelIdx = 0;
                for (auto l : uniqueLabels)
                    labelsMap[l] = labelIdx++;
            }

            std::vector<size_t> numericLabels;
            for (auto l : labels)
                numericLabels.push_back( labelsMap.at( l ));
            return numericLabels;
        }

        virtual void _trainLDA( const std::vector<std::vector<double >> &features,
                                const std::vector<std::string_view> &labels )
        {
            const size_t ncol = features.front().size();
            assert( std::all_of( features.begin(), features.end(),
                                 [=]( auto &v ) { return v.size() == ncol; } ));

            const auto numericLabels = _numericLabels( labels );

            dlib::matrix<double> X;
            dlib::matrix<double, 0, 1> M;
            X.set_size( features.size(), ncol );
            for (size_t r = 0; r < features.size(); ++r)
                for (size_t c = 0; c < ncol; ++c)
                    X( r, c ) = features.at( r ).at( c );

            if ( _ldaDims )
                dlib::compute_lda_transform( X, M, numericLabels, _ldaDims.value());
            else dlib::compute_lda_transform( X, M, numericLabels );

            _Z = std::move( X );
            _M = std::move( M );

        }

        virtual std::vector<FeatureVector> _transformFeatures( std::vector<FeatureVector> &&features ) const
        {
            for (auto &f : features)
                f = _transformFeatures( f );
            return features;
        }

        virtual FeatureVector _transformFeatures( const FeatureVector &f ) const
        {
            assert( _enableLDA && _Z && _M );
            if ( _enableLDA && _Z && _M )
            {
                dlib::matrix<double, 0, 1> fv =
                        _Z.value() * vector_to_cmatrix( f ) - _M.value();

                return std::vector<double>( fv.begin(), fv.end());
            } else return f;
        }

        void _trainNormalizers( const std::vector<std::vector<double >> &features )
        {
            const size_t ncol = features.front().size();
            assert( std::all_of( features.begin(), features.end(),
                                 [=]( auto &v ) { return v.size() == ncol; } ));

            _colMin = std::vector<double>( ncol, inf );
            _colMax = std::vector<double>( ncol, -inf );
            _colMagnitude = std::vector<double>( ncol, 0 );
            _centroid = std::vector<double>( ncol, 0 );

            for (auto &fs : features)
            {
                for (size_t col = 0; col < ncol; ++col)
                {
                    _colMin[col] = std::min( _colMin[col], fs[col] );
                    _colMax[col] = std::max( _colMax[col], fs[col] );
                    _colMagnitude[col] += fs[col] * fs[col];
                    _centroid[col] += fs[col];
                }
            }

            for (auto &m : _colMagnitude)
                m = std::sqrt( m );

            for (auto &m : _centroid)
                m /= features.size();
        }

        std::vector<std::vector<double >>
        _normalizeFeatures( std::vector<std::vector<double> > &&features ) const
        {
            for (auto &f : features)
                f = _normalizer.value()( std::move( f ));
            return features;
        }

        std::vector<double> _normalizeFeatures( std::vector<double> &&features ) const
        {
            return _normalizer.value()( std::move( features ));
        }

        std::vector<double> _centerScaleColumns( std::vector<double> &&features ) const
        {
            assert( features.size() == _colMin.size() && _colMin.size() == _colMax.size()
                    && _centroid.size() == _colMax.size());
            for (size_t col = 0; col < features.size(); ++col)
                features[col] = (features[col] - _centroid[col]) / (_colMax[col] - _colMin[col]);
            return features;
        }

        std::vector<double> _minmaxScaleColumns( std::vector<double> &&features ) const
        {
            assert( features.size() == _colMin.size() && _colMin.size() == _colMax.size());
            for (size_t col = 0; col < features.size(); ++col)
                features[col] = (features[col] - _colMin[col]) / (_colMax[col] - _colMin[col]);
            return features;
        }


        std::vector<double> _minmaxScaleRow( std::vector<double> &&features ) const
        {
            auto[minIt, maxIt] = std::minmax_element( features.cbegin(), features.cend());
            const double min = *minIt;
            const double max = *maxIt;
            for (double &f : features)
                f = (f - min) / (max - min);
            return features;
        }

        std::vector<double> _radiusNormalizeRow( std::vector<double> &&features ) const
        {
            auto[minIt, maxIt] = std::minmax_element( features.cbegin(), features.cend());
            const double min = *minIt;
            const double max = *maxIt;
            double magnitude = std::accumulate( features.cbegin(), features.cend(), double( 0 ),
                                                []( double acc, double val ) {
                                                    return acc + val * val;
                                                } );
            for (double &f : features)
                f /= magnitude;
            return features;
        }

        std::vector<double> _radiusNormalizeColumns( std::vector<double> &&features ) const
        {
            assert( features.size() == _colMagnitude.size());
            for (size_t col = 0; col < features.size(); ++col)
                features[col] = features[col] / (_colMagnitude[col] + eps);
            return features;
        }


    private:
        std::vector<double> _colMin;
        std::vector<double> _colMax;
        std::vector<double> _colMagnitude;
        std::vector<double> _centroid;
        std::optional<NormalizerFunction> _normalizer;


        std::optional<dlib::matrix<double>> _Z;
        std::optional<dlib::matrix<double, 0, 1>> _M;
        std::optional<size_t> _ldaDims;
        bool _enableLDA;
    };
}
#endif //MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
