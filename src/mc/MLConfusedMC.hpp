//
// Created by asem on 13/09/18.
//

#ifndef MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
#define MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP

#include "common.hpp"
#include "AbstractMC.hpp"

namespace MC {
    class MLConfusedMC
    {

    public:
        using FeatureVector = std::vector<double>;

        void fit( const std::map<std::string_view, std::vector<std::string >> &training )
        {
//            _normalizer = [this]( auto &&f ) { return _centerScaleColumns( std::move( f )); };

            std::vector<std::string_view> labels;
            std::vector<FeatureVector> featuresVector;
            for (auto &[trainLabel, trainSeqs] : training)
            {
                for (auto &seq : trainSeqs)
                {
                    if ( auto features = extractFeatures( seq ); features )
                    {
                        assert( std::all_of( features->begin(), features->end(),
                                             []( double v ) { return !std::isnan( v ); } ));

                        featuresVector.emplace_back( features.value());
                        labels.push_back( trainLabel );
                    }

                }
            }
            if ( _normalizer )
            {
                _trainNormalizers( featuresVector );
                featuresVector = _normalizeFeatures( std::move( featuresVector ));
            }
            fitML( labels, std::move( featuresVector ));
        }


        std::string_view predict( const std::string &sequence ) const
        {

            if ( auto features = extractFeatures( sequence ); features )
            {
                assert( std::all_of( features->begin(), features->end(),
                                     []( double v ) { return !std::isnan( v ); } ));

                if ( _normalizer )
                    return predictML( _normalizeFeatures( std::move( features.value())));
                else return predictML( features.value());
            } else return unclassified;
        }

    protected:
        virtual void fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) = 0;

        virtual std::string_view predictML( const FeatureVector &f ) const = 0;

        virtual std::optional<FeatureVector> extractFeatures( std::string_view sequence ) const = 0;

    private:
        using NormalizerFunction = std::function<std::vector<double>( std::vector<double> && )>;

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
            auto[min, max] = std::minmax_element( features.cbegin(), features.cend());

            for (double &f : features)
                f = (f - *min) / (*max - *min);
            return features;
        }

        std::vector<double> _radiusNormalizeRow( std::vector<double> &&features ) const
        {
            auto[min, max] = std::minmax_element( features.cbegin(), features.cend());
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
    };
}
#endif //MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
