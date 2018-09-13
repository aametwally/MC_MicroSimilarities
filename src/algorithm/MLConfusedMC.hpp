//
// Created by asem on 13/09/18.
//

#ifndef MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
#define MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP

#include "common.hpp"
#include "AbstractMC.hpp"

namespace MC {

    const std::string unclassified = "unclassified";

    class MLConfusedMC
    {

    public:
        using FeatureVector = std::vector<double>;

        void fit( const std::map<std::string, std::vector<std::string >> &training )
        {
            _normalizer = [this]( auto &&f ){ return _minmaxScaleColumns( std::move( f) );};

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
            _trainNormalizers( featuresVector );
            featuresVector = _normalizeFeatures( std::move( featuresVector ));
            fitML( labels, std::move( featuresVector ));
        }


        std::string_view predict( const std::string &sequence ) const
        {

            if ( auto features = extractFeatures( sequence ); features )
            {
                assert( std::all_of( features->begin(), features->end(),
                                     []( double v ) { return !std::isnan( v ); } ));


                return predictML( _normalizeFeatures( std::move( features.value())));
            } else return unclassified;
        }

    protected:
        virtual void fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) = 0;

        virtual std::string_view predictML( const FeatureVector &f ) const = 0;

        virtual std::optional<FeatureVector> extractFeatures( const std::string &sequence ) const = 0;

    private:
        using NormalizerFunction = std::function<std::vector<double>( std::vector<double> && )>;

        void _trainNormalizers( const std::vector<std::vector<double >> &features )
        {
            const size_t ncol = features.front().size();
            assert( std::all_of( features.begin(), features.end(),
                                 [=]( auto &v ) { return v.size() == ncol; } ));

            _min = std::vector<double>( ncol, inf );
            _max = std::vector<double>( ncol, -inf );
            _magnitude = std::vector<double>( ncol, 0 );

            for (auto &fs : features)
            {
                for (size_t col = 0; col < ncol; ++col)
                {
                    _min[col] = std::min( _min[col], fs[col] );
                    _max[col] = std::max( _max[col], fs[col] );
                    _magnitude[col] += fs[col] * fs[col];
                }
            }

            for (auto &m : _magnitude)
                m = std::sqrt( m );

        }

        std::vector<std::vector<double >>
        _normalizeFeatures( std::vector<std::vector<double> > &&features ) const
        {
            for (auto &f : features)
                f = _normalizer( std::move( f ));
            return features;
        }

        std::vector<double> _normalizeFeatures( std::vector<double> &&features ) const
        {
            return _normalizer( std::move( features ));
        }

        std::vector<double> _minmaxScaleColumns( std::vector<double> &&features ) const
        {
            assert( features.size() == _min.size() && _min.size() == _max.size());
            for (size_t col = 0; col < features.size(); ++col)
                features[col] = (features[col] - _min[col]) / (_max[col] - _min[col]);
            return features;
        }

        std::vector<double> _radiusNormalizeColumns( std::vector<double> &&features ) const
        {
            assert( features.size() == _magnitude.size());
            for (size_t col = 0; col < features.size(); ++col)
                features[col] = features[col] / (_magnitude[col] + eps);
            return features;
        }

    private:
        std::vector<double> _min;
        std::vector<double> _max;
        std::vector<double> _magnitude;

        NormalizerFunction _normalizer;
    };
}
#endif //MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
