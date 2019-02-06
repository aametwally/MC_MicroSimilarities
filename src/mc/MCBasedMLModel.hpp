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

struct LDAConfiguration
{
    size_t nDim = 0;
};

struct PCAConfiguration
{

};

template<size_t States>
class MCBasedMLModel : public AbstractMCClassifier<States>
{
    static constexpr double eps = std::numeric_limits<double>::epsilon();

    struct LDAData
    {
        /**
         * Given an input vector x, Z*x-M, is the transformed version of x.
         */
        dlib::matrix<double> Z;
        dlib::matrix<double, 0, 1> M;
    };

    struct NormalizationData
    {
        std::vector<double> colMin;
        std::vector<double> colMax;
        std::vector<double> colMagnitude;
        std::vector<double> colStandardDeviation;
        std::vector<double> centroid;
    };

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
    MCBasedMLModel( ModelGenerator <States> generator,
                    std::optional<LDAConfiguration> ldaConfig,
                    std::optional<PCAConfiguration> pcaConfig )
            : AbstractMCClassifier<States>( generator ),
              _ldaConfiguration( std::move( ldaConfig )),
              _pcaConfiguration( std::move( pcaConfig ))
    {}

    virtual ~MCBasedMLModel() = default;


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
//                               new KNNMCMicroSimilarity<Grouping>( backbones, background, training,
//                                                              7, trainer, similarity ));
//
//            _ensemble.emplace( ClassificationEnum::SVM,
//                               new SVMCMMicroSimilarity<Grouping>( backbones, background, training, trainer, similarity ));

//            _ensemble.emplace( ClassificationEnum::KMERS,
//                               new MCKmersClassifier<Grouping>( backbones, background ));
    }

    void fit( const std::map<std::string_view, std::vector<std::string >> &training )
    {
        fit( training, this->_backbones, this->_backgrounds, this->_centralBackground );
    }

    void fit( const std::map<std::string_view, std::vector<std::string >> &training,
              const BackboneProfiles &backbones,
              const BackboneProfiles &backgrounds,
              const BackboneProfile &centralBackground )
    {
        _normalizer = [this]( FeatureVector &&f ) { return _standardNormalizeColumns( std::move( f )); };

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
        if ( _ldaConfiguration )
        {
            _trainLDA( featuresVector, labels );
            featuresVector = _transformFeatures( std::move( featuresVector ));
        }
        _fitML( std::move( labels ), std::move( featuresVector ));
    }

protected:
    ScoredLabels _predict( std::string_view sequence,
                           const BackboneProfiles &backboneProfiles,
                           const BackboneProfiles &backgroundProfiles,
                           const BackboneProfile &centralBackground ) const override
    {
        auto features = _extractFeatures( sequence, backboneProfiles, backgroundProfiles, centralBackground );
        assert( std::all_of( features.begin(), features.end(),
                             []( double v ) { return !std::isnan( v ); } ));

        if ( _normalizer )
            features = _normalizeFeatures( std::move( features ));

        assert((_ldaConfiguration && _ldaData) || !_ldaConfiguration );
        if ( _ldaConfiguration && _ldaData )
            features = _transformFeatures( std::move( features ));

        return _predictML( std::move( features ));
    }

    virtual void _fitML( std::vector<std::string_view> &&labels, std::vector<FeatureVector> &&f ) = 0;

    virtual ScoredLabels _predictML( FeatureVector &&f ) const = 0;

    virtual FeatureVector _extractFeatures( std::string_view sequence,
                                            const BackboneProfiles &backboneProfiles,
                                            const BackboneProfiles &backgroundProfiles,
                                            const BackboneProfile &centralBackground ) const
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
                             [=]( auto &v ) { return v.size() == ncol; } ) && _ldaConfiguration );

        const auto numericLabels = _numericLabels( labels );

        dlib::matrix<double> X;
        dlib::matrix<double, 0, 1> M;
        X.set_size( features.size(), ncol );
        for (size_t r = 0; r < features.size(); ++r)
            for (size_t c = 0; c < ncol; ++c)
                X( r, c ) = features.at( r ).at( c );

        if ( _ldaConfiguration && _ldaConfiguration->nDim > 0 )
            dlib::compute_lda_transform( X, M, numericLabels, _ldaConfiguration->nDim );
        else dlib::compute_lda_transform( X, M, numericLabels );

        _ldaData.emplace();
        _ldaData->Z = std::move( X );
        _ldaData->M = std::move( M );
    }

    virtual std::vector<FeatureVector> _transformFeatures( std::vector<FeatureVector> &&features ) const
    {
        for (auto &&f : features)
            f = _transformFeatures( std::move( f ));
        return features;
    }

    virtual FeatureVector _transformFeatures( FeatureVector &&f ) const
    {
        using namespace dlib_utilities;
        assert( _ldaConfiguration && _ldaData );
        if ( _ldaConfiguration && _ldaData )
        {
            auto &&Z = _ldaData->Z;
            auto &&M = _ldaData->M;
            auto x = vectorToColumnMatrixLike( std::move( f ));
            ColumnVectorMatrixLike<double> fv( Z * x - M );
            return std::move( fv.steal_vector());
        } else return f;
    }

    void _trainNormalizers( const std::vector<std::vector<double >> &features )
    {
        const size_t nrow = features.size();
        const size_t ncol = features.front().size();
        assert( std::all_of( features.cbegin(), features.cend(),
                             [=]( auto &&v ) { return v.size() == ncol; } ));

        _normData.emplace();
        auto &&colMin = _normData->colMin;
        auto &&colMax = _normData->colMax;
        auto &&colMagnitude = _normData->colMagnitude;
        auto &&centroid = _normData->centroid;
        auto &&colStandardDeviation = _normData->colStandardDeviation;

        colMin = std::vector<double>( ncol, inf );
        colMax = std::vector<double>( ncol, -inf );
        colMagnitude = std::vector<double>( ncol, 0 );
        colStandardDeviation = std::vector<double>( ncol, 0 );
        centroid = std::vector<double>( ncol, 0 );

        size_t n = 0;
        for (auto &&fs : features)
        {
            for (size_t col = 0; col < ncol; ++col)
            {
                if ( !std::isnan( fs[col] ))
                {
                    colMin[col] = std::min( colMin[col], fs[col] );
                    colMax[col] = std::max( colMax[col], fs[col] );
                    colMagnitude[col] += fs[col] * fs[col];
                    centroid[col] += fs[col];
                    ++n;
                }
            }
        }

        for (auto &&m : colMagnitude)
            m = std::sqrt( m );

        for (auto &&m : centroid)
            m /= n;

        for (auto &&fs : features)
        {
            for (size_t col = 0; col < ncol; ++col)
            {
                if ( !std::isnan( fs[col] ))
                {
                    auto term = centroid[col] - fs[col];
                    colStandardDeviation[col] += term * term;
                }
            }
        }

        for (auto &&s : colStandardDeviation)
            s = std::sqrt( s / n );
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
        auto &&centroid = _normData->centroid;
        auto &&colMax = _normData->colMax;
        auto &&colMin = _normData->colMin;

        assert( features.size() == colMin.size() && colMin.size() == colMax.size()
                && centroid.size() == colMax.size());
        for (size_t col = 0; col < features.size(); ++col)
        {
            features[col] = (features[col] - centroid[col]) / (colMax[col] - colMin[col] + eps);
        }
        return features;
    }

    std::vector<double> _minmaxScaleColumns( std::vector<double> &&features ) const
    {
        auto &&colMax = _normData->colMax;
        auto &&colMin = _normData->colMin;

        assert( features.size() == colMin.size() && colMin.size() == colMax.size());
        for (size_t col = 0; col < features.size(); ++col)
        {
            features[col] = (features[col] - colMin[col]) / (colMax[col] - colMin[col] + eps);
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
        auto &&colMagnitude = _normData->colMagnitude;

        assert( features.size() == colMagnitude.size());
        for (size_t col = 0; col < features.size(); ++col)
            features[col] = features[col] / (colMagnitude[col] + eps);
        return features;
    }

    std::vector<double> _standardNormalizeColumns( std::vector<double> &&features ) const
    {
        auto &&colStandardDeviation = _normData->colStandardDeviation;
        auto &&centroid = _normData->centroid;

        assert( features.size() == colStandardDeviation.size());
        for (size_t col = 0; col < features.size(); ++col)
            features[col] = (features[col] - centroid[col]) / (colStandardDeviation[col] + eps);
        return features;
    }

protected:
    std::map<ClassificationEnum, std::unique_ptr<AbstractMCClassifier < States> >> _ensemble;

private:
    std::optional<NormalizationData> _normData;
    std::optional<NormalizerFunction> _normalizer;
    std::optional<LDAConfiguration> _ldaConfiguration;
    std::optional<PCAConfiguration> _pcaConfiguration;
    std::optional<LDAData> _ldaData;
};

template<size_t States>
class KNNStackedMC : protected KNNModel<Euclidean>, public MCBasedMLModel<States>
{
    using KNN = KNNModel<Euclidean>;
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using Generator = ModelGenerator<States>;

public:
    explicit KNNStackedMC(
            size_t k, Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : KNN( k ), MCBasedMLModel<States>( generator, ldaConfig, pcaConfig )
    {}

    virtual ~KNNStackedMC() = default;

    using MCBasedMLModel<States>::fit;

protected:
    void _fitML( std::vector<std::string_view> &&labels, std::vector<FeatureVector> &&f ) override
    {
        KNN::fit( std::move( labels), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return KNN::predict( std::move( f ));
    }
};

template<size_t States>
class SVMStackedMC : protected SVMModel, public MCBasedMLModel<States>
{
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using Generator = ModelGenerator<States>;

public:
    explicit SVMStackedMC(
            SVMConfiguration configuration,
            const Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : MCBasedMLModel<States>( generator, ldaConfig, pcaConfig ),
              SVMModel( configuration )
    {}

    virtual ~SVMStackedMC() = default;

    using MCBasedMLModel<States>::fit;
protected:
    void _fitML( std::vector<std::string_view> &&labels, std::vector<FeatureVector> &&f ) override
    {
        SVMModel::fit( std::move( labels), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return SVMModel::predict( std::move( f ));
    }
};

template<size_t States>
class MCParametersBasedMLClassifier : public MCBasedMLModel<States>
{
public:
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using BackboneProfile = typename MCModel::BackboneProfile;
    using Generator = ModelGenerator<States>;

public:
    explicit MCParametersBasedMLClassifier(
            Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : MCBasedMLModel<States>( generator, ldaConfig, pcaConfig )
    {}

    void initWeakModels( SimilarityFunctor<Histogram> similarity ) override
    {}

protected:
    FeatureVector _extractFeatures( std::string_view sequence,
                                    const BackboneProfiles &backboneProfiles,
                                    const BackboneProfiles &backgroundProfiles,
                                    const BackboneProfile &centralBackground ) const override
    {
        assert( centralBackground ); // must exist.

        size_t nCols = std::accumulate( backboneProfiles.cbegin(), backboneProfiles.cend(), size_t( 0 ),
                                        []( size_t acc, auto &&kv ) {
                                            return acc + kv.second.parametersCount();
                                        } );
        std::vector<double> flatFeatures;
        flatFeatures.reserve( nCols ); // TODO: handle exceptions.


        auto sample = this->_generator( sequence );
        std::map<std::string_view, std::vector<double >> similarities;
        for (auto &[label, profile] : backboneProfiles)
        {
            auto &&backboneCentroids = profile->centroids().get();
            backboneCentroids.forEach( [&]( Order order, HistogramID id, auto &&backboneCentroid ) {

                auto sampleCentroid = sample->centroid( order, id );
                if ( sampleCentroid )
                {
                    // auto standardDeviation = profile->standardDeviation( order, id );
                    flatFeatures.insert(
                            flatFeatures.end(), sampleCentroid->get().cbegin(), sampleCentroid->get().cend());

                } else if ( centralBackground )
                {
                    if ( auto center = centralBackground->centroid( order, id ); center.has_value())
                    {
                        //auto standardDeviation = profile->standardDeviation( order, id );
                        flatFeatures.insert(
                                flatFeatures.end(), center->get().cbegin(), center->get().cend());
                    }
                } else
                {
                    throw std::runtime_error( "Unhandled case!" );
                }
            } );
        }
        return flatFeatures;
    }

private:
};

template<size_t States>
class SVMMCParameters : private SVMModel, public MCParametersBasedMLClassifier<States>
{
public:
    using FeatureVector = typename MCParametersBasedMLClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit SVMMCParameters(
            SVMConfiguration configuration,
            Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : SVMModel( configuration ),
              MCParametersBasedMLClassifier<States>( generator, ldaConfig, pcaConfig )
    {}

    virtual ~SVMMCParameters() = default;

    using MCParametersBasedMLClassifier<States>::fit;

protected:
    void _fitML( std::vector<std::string_view> &&labels, std::vector<FeatureVector> &&f ) override
    {
        SVMModel::fit( std::move( labels), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return SVMModel::predict( std::move( f ));
    }
};

template<size_t States>
class KNNMCParameters : private KNNModel<Euclidean>, public MCParametersBasedMLClassifier<States>
{
public:
    using FeatureVector = typename MCParametersBasedMLClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit KNNMCParameters(
            size_t k,
            Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : KNNModel( k ),
              MCParametersBasedMLClassifier<States>( generator, ldaConfig, pcaConfig )
    {}

    virtual ~KNNMCParameters() = default;

    using MCParametersBasedMLClassifier<States>::fit;

protected:
    void _fitML( std::vector<std::string_view> &&labels, std::vector<FeatureVector> &&f ) override
    {
        KNNModel::fit( std::move( labels), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return KNNModel::predict( std::move( f ));
    }
};

template<size_t States>
class MCMicroSimilarityBasedMLClassifier : public MCBasedMLModel<States>
{
public:
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using BackboneProfile = typename MCModel::BackboneProfile;
    using Generator = ModelGenerator<States>;
    using MicroMeasurements = std::map<std::string_view, std::unordered_map<Order, std::unordered_map<HistogramID, double >>>;
    using AlternativeMeasurements = std::unordered_map<Order, std::unordered_map<HistogramID, double >>;

public:
    explicit MCMicroSimilarityBasedMLClassifier(
            Generator generator,
            SimilarityFunctor<Histogram> similarity,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : MCBasedMLModel<States>( generator, ldaConfig, pcaConfig ),
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
                                    const BackboneProfile &centralBackground ) const override
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
                double &furthest = alternatives[order].try_emplace( id, bestInfinity ).first->second;

                auto sampleCentroid = sample->centroid( order, id );
                if ( sampleCentroid )
                {
                    auto standardDeviation = profile->standardDeviation( order, id );
                    measurement = this->_similarity( backboneCentroid, sampleCentroid->get(),
                                                     standardDeviation->get());
                } else if ( centralBackground )
                {
                    if ( auto center = centralBackground->centroid( order, id ); center.has_value())
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

//        measurements = compensateMissingMeasurements( std::move( measurements ), std::move( alternatives ));
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
class SVMCMMicroSimilarity : private SVMModel, public MCMicroSimilarityBasedMLClassifier<States>
{
public:
    using FeatureVector = typename MCMicroSimilarityBasedMLClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit SVMCMMicroSimilarity(
            SVMConfiguration configuration,
            Generator generator,
            SimilarityFunctor<Histogram> similarity,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : SVMModel( configuration ),
              MCMicroSimilarityBasedMLClassifier<States>( generator, similarity, ldaConfig, pcaConfig )
    {}

    virtual ~SVMCMMicroSimilarity() = default;

    using MCMicroSimilarityBasedMLClassifier<States>::fit;

protected:
    void _fitML( std::vector<std::string_view> &&labels, std::vector<FeatureVector> &&f ) override
    {
        SVMModel::fit( std::move( labels), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return SVMModel::predict( std::move( f ));
    }
};

template<size_t States>
class KNNMCMicroSimilarity : private KNNModel<Euclidean>, public MCMicroSimilarityBasedMLClassifier<States>
{
public:
    using FeatureVector = typename MCMicroSimilarityBasedMLClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit KNNMCMicroSimilarity(
            size_t k,
            Generator generator,
            SimilarityFunctor<Histogram> similarity,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<PCAConfiguration> pcaConfig )
            : KNNModel( k ),
              MCMicroSimilarityBasedMLClassifier<States>( generator, similarity, ldaConfig, pcaConfig )
    {}

    virtual ~KNNMCMicroSimilarity() = default;

    using MCMicroSimilarityBasedMLClassifier<States>::fit;

protected:
    void _fitML( std::vector<std::string_view> &&labels, std::vector<FeatureVector> &&f ) override
    {
        KNNModel::fit( std::move( labels), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return KNNModel::predict( std::move( f ));
    }
};

}
#endif //MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
