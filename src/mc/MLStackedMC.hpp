//
// Created by asem on 13/09/18.
//

#ifndef MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
#define MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP

#include "common.hpp"
#include "AbstractMC.hpp"
#include "AbstractMCClassifier.hpp"

#include "KNNModel.hpp"
#include "SVMModel.hpp"

#include "dlib_utilities.hpp"
#include "dlib/svm.h"


namespace MC {

template<size_t States>
class MLStackedMC : public AbstractMCClassifier<States>
{
    static constexpr double eps = std::numeric_limits<double>::epsilon();

public:
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using MCF = MCFeatures<States>;
    using BackboneProfile = typename MCModel::BackboneProfile;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using ModelTrainer =  ModelGenerator<States>;
    using MacroSimilarityEnum = typename MicroSimilarityBasedClassifier<States>::MacroScoringEnum;
    using FeatureVector = std::vector<double>;

public:
    MLStackedMC( ModelGenerator <States> generator )
            : AbstractMCClassifier<States>( generator ),
              _enableLDA( false )
    {}

    virtual ~MLStackedMC() = default;


    virtual void initWeakModels( SimilarityFunctor<Histogram> similarity )
    {
        _ensemble.emplace(
                ClassificationEnum::Propensity, new MCPropensityClassifier<States>( this->_generator ));

        _ensemble.emplace(
                ClassificationEnum::MacroSimilarityAccumulative,
                new MicroSimilarityBasedClassifier<States>( MacroSimilarityEnum::Accumulative,
                                                            this->_generator, similarity ));

        _ensemble.emplace(
                ClassificationEnum::MacroSimilarityVoting,
                new MicroSimilarityBasedClassifier<States>( MacroSimilarityEnum::Voting,
                                                            this->_generator, similarity ));


//            _ensemble.emplace( ClassificationEnum::KNN,
//                               new KNNMCParameters<Grouping>( backbones, background, training,
//                                                              7, trainer, similarity ));
//
//            _ensemble.emplace( ClassificationEnum::SVM,
//                               new SVMMCParameters<Grouping>( backbones, background, training, trainer, similarity ));

//            _ensemble.emplace( ClassificationEnum::KMERS,
//                               new MCKmersClassifier<Grouping>( backbones, background ));
    }

    void fit( const std::map<std::string_view, std::vector<std::string >> &training )
    {
        enableLDA();
        fit( training, this->_backbones, this->_backgrounds, this->_centralBackground );
    }

    void enableLDA( std::optional<size_t> ldaDims = std::nullopt )
    {
        _enableLDA = true;
        _ldaDims = ldaDims;
    }

    void fit( const std::map<std::string_view, std::vector<std::string >> &training,
              const BackboneProfiles &backbones,
              const BackboneProfiles &backgrounds,
              const std::optional<BackboneProfile> &centralBackground )
    {
        _normalizer = [this]( auto &&f ) { return _minmaxScaleColumns( std::move( f )); };

        std::vector<std::string_view> labels;
        std::vector<FeatureVector> featuresVector;
        for (auto &[trainLabel, trainSeqs] : training)
        {
            for (auto &&seq : trainSeqs)
            {
                auto features = _extractFeatures( seq, backbones, backgrounds, centralBackground );
                assert( std::all_of( features.begin(), features.end(),
                                     []( double v ) { return !std::isnan( v ); } ));
                featuresVector.emplace_back( features );
                labels.push_back( trainLabel );
            }
        }

        assert( [&]() {
            const size_t n = featuresVector.front().size();
            return std::all_of( featuresVector.cbegin(), featuresVector.cend(),
                                [=]( auto &&v ) {
                                    return v.size() == n;
                                } );
        }());

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
    ScoredLabels _predict( std::string_view sequence,
                           const BackboneProfiles &backboneProfiles,
                           const BackboneProfiles &backgroundProfiles,
                           const std::optional<BackboneProfile> &centralBackground ) const override
    {
        auto features = _extractFeatures( sequence, backboneProfiles, backgroundProfiles, centralBackground );
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

    virtual FeatureVector _extractFeatures( std::string_view sequence,
                                            const BackboneProfiles &backboneProfiles,
                                            const BackboneProfiles &backgroundProfiles,
                                            const std::optional<BackboneProfile> &centralBackground ) const
    {
        FeatureVector f;
        for (auto &[enumm, classifier] : this->_ensemble)
        {
            auto propensityPredictions =
                    classifier->scoredPredictions(
                            sequence,
                            backboneProfiles,
                            backgroundProfiles,
                            centralBackground ).toMap();

            for (auto &[cluster, _] : backboneProfiles)
                f.push_back( propensityPredictions.at( cluster ));
        }
        f.push_back( sequence.length());
        return f;
    };

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
        for (auto &&l : labels)
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
        for (auto &&f : features)
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
        assert( std::all_of( features.cbegin(), features.cend(),
                             [=]( auto &&v ) { return v.size() == ncol; } ));

        _colMin = std::vector<double>( ncol, inf );
        _colMax = std::vector<double>( ncol, -inf );
        _colMagnitude = std::vector<double>( ncol, 0 );
        _centroid = std::vector<double>( ncol, 0 );

        for (auto &&fs : features)
        {
            for (size_t col = 0; col < ncol; ++col)
            {
                _colMin[col] = std::min( _colMin[col], fs[col] );
                _colMax[col] = std::max( _colMax[col], fs[col] );
                _colMagnitude[col] += fs[col] * fs[col];
                _centroid[col] += fs[col];
            }
        }

        for (auto &&m : _colMagnitude)
            m = std::sqrt( m );

        for (auto &&m : _centroid)
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
            features[col] = (features[col] - _centroid[col]) / (_colMax[col] - _colMin[col] + eps);
        return features;
    }

    std::vector<double> _minmaxScaleColumns( std::vector<double> &&features ) const
    {
        assert( features.size() == _colMin.size() && _colMin.size() == _colMax.size());
        for (size_t col = 0; col < features.size(); ++col)
        {
            features[col] = (features[col] - _colMin[col]) / (_colMax[col] - _colMin[col] + eps);
        }
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

protected:
    std::map<ClassificationEnum, std::unique_ptr<AbstractMCClassifier < States> >> _ensemble;

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


template<size_t States>
class KNNStackedMC : protected KNNModel<Euclidean>, public MLStackedMC<States>
{
    using KNN = KNNModel<Euclidean>;
    using FeatureVector = typename MLStackedMC<States>::FeatureVector;
    using Generator = ModelGenerator<States>;

public:
    explicit KNNStackedMC( Generator generator, size_t k )
            : KNN( k ), MLStackedMC<States>( generator )
    {}

    virtual ~KNNStackedMC() = default;

    using MLStackedMC<States>::fit;

protected:
    void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
    {
        KNN::fit( labels, std::move( f ));
    }

    ScoredLabels _predictML( const FeatureVector &f ) const override
    {
        assert( _validTraining());
        return KNN::predict( f );
    }
};

template<size_t States>
class SVMStackedMC : protected SVMModel, public MLStackedMC<States>
{
    using FeatureVector = typename MLStackedMC<States>::FeatureVector;
    using Generator = ModelGenerator<States>;

public:
    explicit SVMStackedMC(
            const Generator generator,
            std::optional<double> lambda = 1,
            std::optional<double> gamma = 10 )
            : MLStackedMC<States>( generator ),
              SVMModel( lambda, gamma )
    {}

    virtual ~SVMStackedMC() = default;

    using MLStackedMC<States>::fit;
protected:
    void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
    {
        SVMModel::fit( labels, std::move( f ));
    }

    ScoredLabels _predictML( const FeatureVector &f ) const override
    {
        assert( _validTraining());
        return SVMModel::predict( f );
    }
};

template<size_t States>
class MCParametersClassifier : public MLStackedMC<States>
{
public:
    using FeatureVector = typename MLStackedMC<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using BackboneProfile = typename MCModel::BackboneProfile;
    using Generator = ModelGenerator<States>;
    using MicroMeasurements = std::map<std::string_view, std::unordered_map<Order, std::unordered_map<HistogramID, double >>>;
    using AlternativeMeasurements = std::unordered_map<Order, std::unordered_map<HistogramID, double >>;

public:
    explicit MCParametersClassifier(
            Generator modelGenerator,
            SimilarityFunctor<Histogram> similarity )
            : MLStackedMC<States>( modelGenerator ),
              _similarity( similarity )
    {}

    void initWeakModels( SimilarityFunctor<Histogram> similarity ) override
    {}

protected:
    template<typename MicroMeasurementsType, typename AlternativeMeasurementsType>
    static MicroMeasurements compensateMissingMeasurements(
            MicroMeasurementsType &&microMeasurements, AlternativeMeasurementsType &&alternativeMeasurements )
    {
        MicroMeasurements compensated = std::forward<MicroMeasurementsType>( microMeasurements );
        for (auto &&[label, measurements] : microMeasurements)
        {
            for (auto &&[order, isoMeasurements] : measurements)
            {
                auto &isoAlternativeMeasurements = alternativeMeasurements.at( order );
                for (auto &&[id, measurement] : isoMeasurements)
                {
                    if ( std::isnan( measurement ))
                    {
                        measurement = isoAlternativeMeasurements.at( id );
                    }
                }
            }
        }
        return compensated;
    }


    FeatureVector _extractFeatures( std::string_view sequence,
                                    const BackboneProfiles &backboneProfiles,
                                    const BackboneProfiles &backgroundProfiles,
                                    const std::optional<BackboneProfile> &centralBackground ) const override
    {
        MicroMeasurements measurements;
        AlternativeMeasurements alternatives;
        auto closerThan = this->_similarity.closerThan;
        auto bestInfinity = this->_similarity.best;

        auto sample = this->_generator( sequence );
        std::map<std::string_view, std::vector<double >> similarities;
        for (auto &[label, profile] : backboneProfiles)
        {
            auto &_measurements = measurements[label];
            auto &&backboneCentroids = profile->centroids().get();
            backboneCentroids.forEach( [&]( Order order, HistogramID id, auto &&backboneCentroid ) {
                double &measurement = _measurements[order][id];
                double &furthest = alternatives[order].try_emplace( id, -inf ).first->second;

                auto sampleCentroid = sample->centroid( order, id );
                if ( sampleCentroid )
                {
                    auto standardDeviation = profile->standardDeviation( order, id );
                    measurement = this->_similarity( backboneCentroid , sampleCentroid->get(),
                                                     standardDeviation->get());
                } else if ( centralBackground )
                {
                    if ( auto center = centralBackground.value()->centroid( order, id ); center.has_value())
                    {
                        auto standardDeviation = profile->standardDeviation( order, id );
                        measurement = this->_similarity( backboneCentroid, center->get(),
                                                         standardDeviation->get());
                    }
                } else
                {
                    measurement = nan;
                }
                if ( !std::isnan( measurement ))
                {
                    if ( closerThan( furthest, measurement ))
                        furthest = measurement;
                }
            } );
        }

        measurements = compensateMissingMeasurements( std::move( measurements ), std::move( alternatives ));
        size_t size = std::accumulate( measurements.cbegin(), measurements.cend(), size_t( 0 ),
                                       []( size_t acc, auto &&kv ) {
                                           return std::accumulate( kv.second.cbegin(), kv.second.cend(), acc,
                                                                   []( size_t acc, auto &&kv ) {
                                                                       return acc + kv.second.size();
                                                                   } );
                                       } );

        std::vector<double> flatFeatures;
        flatFeatures.reserve( size );

        for (auto &&[_, classMeasurements] : measurements)
            for (auto &&[_, isoMeasurements]: classMeasurements)
                for (auto &&[_, measurement]: isoMeasurements)
                    flatFeatures.push_back( measurement );

        return flatFeatures;
    }

private:
    const SimilarityFunctor<Histogram> _similarity;
};

template<size_t States>
class SVMMCParameters : private SVMModel, public MCParametersClassifier<States>
{
public:
    using FeatureVector = typename MCParametersClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit SVMMCParameters(
            Generator modelGenerator,
            SimilarityFunctor<Histogram> similarity,
            std::optional<double> lambda = 1,
            std::optional<double> gamma = 10 )
            : MCParametersClassifier<States>( modelGenerator, similarity ),
              SVMModel( lambda, gamma )
    {}

    virtual ~SVMMCParameters() = default;

    using MCParametersClassifier<States>::fit;

protected:
    void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
    {
        SVMModel::fit( labels, std::move( f ));
    }

    ScoredLabels _predictML( const FeatureVector &f ) const override
    {
        return SVMModel::predict( f );
    }
};

template<size_t States>
class KNNMCParameters : private KNNModel<Euclidean>, public MCParametersClassifier<States>
{
public:
    using FeatureVector = typename MCParametersClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit KNNMCParameters(
            Generator modelGenerator,
            SimilarityFunctor<Histogram> similarity,
            size_t k )
            : MCParametersClassifier<States>( modelGenerator, similarity ),
              KNNModel( k )
    {}

    virtual ~KNNMCParameters() = default;

    using MCParametersClassifier<States>::fit;

protected:
    void _fitML( const std::vector<std::string_view> &labels, std::vector<FeatureVector> &&f ) override
    {
        KNNModel::fit( labels, std::move( f ));
    }

    ScoredLabels _predictML( const FeatureVector &f ) const override
    {
        return KNNModel::predict( f );
    }
};

}
#endif //MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
