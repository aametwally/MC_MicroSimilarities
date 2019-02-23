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
#include "RandomForestModel.hpp"

#include "dlib_utilities.hpp"
#include "dlib/svm.h"
#include "dlib/statistics.h"

namespace MC
{

struct LDAConfiguration
{
    size_t nDim = 0;
};

struct SVDConfiguration
{
    double explainedVariance = 0.99;
};

enum class NaNsHandlingEnum
{
    None,
    Neutralize,
    WorstCase,
    Median
};

enum class NormalizationEnum
{
    None,
    ZScores,
    MinMaxScale
};

template < size_t States >
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

    struct SVDData
    {
        /**
         * Given an input vector x, (x-M) * Vk, is the transformed version of x.
         * V == tr(VT) [nFeatures * nFeatures]
         * Vk is a subsetted columns from V, that correspont to the most significant PC.
         */
        dlib::matrix<double> Vk;
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
    using FeatureVector = std::vector<double>;

public:
    MCBasedMLModel(
            ModelGenerator <States> generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : AbstractMCClassifier<States>( generator ),
              _ldaConfiguration( std::move( ldaConfig )),
              _svdConfiguration( std::move( pcaConfig ))
    {
        setNormalization( NormalizationEnum::ZScores );
        setNaNsHandling( NaNsHandlingEnum::None );
    }

    virtual ~MCBasedMLModel() = default;

    void fit( const std::map<std::string_view, std::vector<std::string >> &training )
    {
        fit( training, this->_backbones, this->_backgrounds, this->_centralBackground );
    }

    void fit(
            const std::map<std::string_view, std::vector<std::string >> &training,
            const BackboneProfiles &backbones,
            const BackboneProfiles &backgrounds,
            const BackboneProfile &centralBackground
    )
    {
        std::vector<std::string_view> labels;
        std::vector<FeatureVector> featuresVector;
        for ( auto &[trainLabel, trainSeqs] : training )
        {
            for ( auto &&seq : trainSeqs )
            {
                auto features = _extractFeatures( seq, backbones, backgrounds, centralBackground );
                assert( std::all_of( features.begin(), features.end(),
                                     []( double v ) { return !std::isnan( v ); } ));
                featuresVector.emplace_back( features );
                labels.push_back( trainLabel );
            }
        }

        assert( [&]()
                {
                  const size_t n = featuresVector.front().size();
                  return std::all_of( featuresVector.cbegin(), featuresVector.cend(),
                                      [=]( auto &&v )
                                      {
                                        return v.size() == n;
                                      } );
                }());


        fmt::print( "Preprocessing ..\n" );
        fmt::print( "Input Matrix Size (n,p): ({},{})\n", featuresVector.size(), featuresVector.front().size());

        if( _nansHandler || _normalizer )
        {
            _trainNormalizers( featuresVector );
        }

        if ( _nansHandler )
        {
            featuresVector = _handleNans( std::move( featuresVector ));
        }
        if ( _normalizer )
        {
            featuresVector = _normalizeFeatures( std::move( featuresVector ));
        }


        if ( _svdConfiguration )
        {
            fmt::print( "Before SVD (n,p): ({},{})\n", featuresVector.size(), featuresVector.front().size());
            if ( _normData ) _trainSVD( featuresVector, _normData->centroid );
            else _trainSVD( featuresVector, std::nullopt );
            featuresVector = _transformFeaturesSVD( std::move( featuresVector ));
            fmt::print( "After SVD (n,p): ({},{})\n", featuresVector.size(), featuresVector.front().size());
        }
        if ( _ldaConfiguration )
        {
            fmt::print( "Before LDA (n,p): ({},{})\n", featuresVector.size(), featuresVector.front().size());
            _trainLDA( featuresVector, labels );
            featuresVector = _transformFeaturesLDA( std::move( featuresVector ));
            fmt::print( "After LDA (n,p): ({},{})\n", featuresVector.size(), featuresVector.front().size());
        }
        fmt::print( "[DONE] Preprocessing ..\n" );
        fmt::print( "Output Matrix Size (n,p): ({},{})\n", featuresVector.size(), featuresVector.front().size());

        fmt::print( "Training ..\n" );
        _fitML( std::move( labels ), std::move( featuresVector ));
        fmt::print( "[DONE] Training ..\n" );
    }

    void setNormalization( NormalizationEnum normalizationLabel )
    {
        switch ( normalizationLabel )
        {
            case NormalizationEnum::ZScores:
            {
                _normalizer = [this]( FeatureVector &&f ) { return _standardNormalizeColumns( std::move( f )); };
            }
                break;
            case NormalizationEnum::MinMaxScale:
            {
                _normalizer = [this]( FeatureVector &&f ) { return _minmaxScaleColumns( std::move( f )); };
            }
                break;
            case NormalizationEnum::None:
            {
                _normalizer.reset();
            }
                break;
            default:
            {
                throw std::runtime_error( "Not implemented normalization settings." );
            }
        }
    }

    void setNaNsHandling( NaNsHandlingEnum nansHandlingLabel )
    {
        switch ( nansHandlingLabel )
        {
            case NaNsHandlingEnum::None:
            {
                _nansHandler.reset();
            }
                break;
            case NaNsHandlingEnum::Neutralize:
            {
                _nansHandler = [this]( FeatureVector &&f )
                {
                  return _handleNans( std::move( f ), _normData->centroid );
                };
            }
                break;
            default:
            {
                throw std::runtime_error( "Not implemented NaNs handling setting." );
            }
        }
    }

protected:
    ScoredLabels _predict(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const override
    {
        auto features = _extractFeatures( sequence, backboneProfiles, backgroundProfiles, centralBackground );
        assert( std::all_of( features.begin(), features.end(),
                             []( double v ) { return !std::isnan( v ); } ));

        if ( _nansHandler )
            features = _handleNans( std::move( features ));

        if ( _normalizer )
            features = _normalizeFeatures( std::move( features ));

        assert(( _svdConfiguration && _svdData ) || !_svdConfiguration );
        if ( _svdConfiguration && _svdData )
            features = _transformFeaturesSVD( std::move( features ));

        assert(( _ldaConfiguration && _ldaData ) || !_ldaConfiguration );
        if ( _ldaConfiguration && _ldaData )
            features = _transformFeaturesLDA( std::move( features ));

        return _predictML( std::move( features ));
    }

    virtual void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) = 0;

    virtual ScoredLabels _predictML( FeatureVector &&f ) const = 0;

    virtual FeatureVector _extractFeatures(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const = 0;

private:
    using NormalizerFunction = std::function<std::vector<double>( std::vector<double> && )>;
    using NaNsHandler = std::function<std::vector<double>( std::vector<double> && )>;

    static std::vector<size_t> _numericLabels( const std::vector<std::string_view> &labels )
    {
        std::set<std::string_view> uniqueLabels( labels.cbegin(), labels.cend());
        std::map<std::string_view, size_t> labelsMap;
        {
            size_t labelIdx = 0;
            for ( auto l : uniqueLabels )
                labelsMap[l] = labelIdx++;
        }

        std::vector<size_t> numericLabels;
        for ( auto &&l : labels )
            numericLabels.push_back( labelsMap.at( l ));
        return numericLabels;
    }

    virtual void _trainLDA(
            const std::vector<std::vector<double >> &features,
            const std::vector<std::string_view> &labels
    )
    {
        const size_t ncol = features.front().size();
        assert( std::all_of( features.begin(), features.end(),
                             [=]( auto &v ) { return v.size() == ncol; } ) && _ldaConfiguration );

        const auto numericLabels = _numericLabels( labels );

        dlib::matrix<double> X;
        dlib::matrix<double, 0, 1> M;
        X.set_size( features.size(), ncol );
        for ( size_t r = 0; r < features.size(); ++r )
            for ( size_t c = 0; c < ncol; ++c )
                X( r, c ) = features.at( r ).at( c );

        if ( _ldaConfiguration && _ldaConfiguration->nDim > 0 )
            dlib::compute_lda_transform( X, M, numericLabels, _ldaConfiguration->nDim );
        else dlib::compute_lda_transform( X, M, numericLabels );

        _ldaData.emplace();
        _ldaData->Z = std::move( X );
        _ldaData->M = std::move( M );
    }

    virtual void _trainSVD(
            const std::vector<std::vector<double >> &features,
            const std::optional<std::vector<double>> &centroid
    )
    {
        const size_t nFeatures = features.front().size();
        const size_t nSamples = features.size();
        assert( std::all_of( features.begin(), features.end(),
                             [=]( auto &v ) { return v.size() == nFeatures; } ) && _svdConfiguration );

        dlib::matrix<double, 1, 0> center =
                ( centroid ) ?
                dlib_utilities::vector_to_column_matrix_like( centroid.value()) :
                dlib_utilities::vector_to_column_matrix_like( _centroid( features ));

        dlib::matrix<double> X, U, S, V; // Data matrix,

        X.set_size( nSamples, nFeatures );
        for ( size_t r = 0; r < X.nr(); ++r )
            for ( size_t c = 0; c < X.nc(); ++c )
                X( r, c ) = features[r][c] - center( c );

        _svdData.emplace();

        dlib::svd3( X, U, S, V );
        fmt::print( "X ({},{}) = DECOMP(U ({},{}), S ({},{}), V ({},{}))\n",
                    X.nr(), X.nc(), U.nr(), U.nc(), S.nr(), S.nc(), V.nr(), V.nc());

        _svdData->Vk = explainedVarianceColumnsSelection( V, S, _svdConfiguration->explainedVariance );
        fmt::print( "Reduced V ({},{}) ==(exp_var:{})==> ({},{})\n",
                    V.nr(), V.nc(), _svdConfiguration->explainedVariance,
                    _svdData->Vk.nr(), _svdData->Vk.nc());

        _svdData->M = std::move( center );
    }

    static dlib::matrix<double> explainedVarianceColumnsSelection(
            const dlib::matrix<double> &V,
            const dlib::matrix<double> &S,
            double exp
    )
    {
        assert( S.nc() == 1 && S.nr() == V.nr() && V.nr() == V.nc());
        assert( exp > 0 && exp <= 1 );
        if ( exp <= 0 )
        {
            throw std::runtime_error( "Bad explained variance selection." );
        } else if ( exp >= 1 )
        {
            return V;
        } else
        {
            std::vector<size_t> indices;
            indices.reserve( static_cast<size_t>(S.nr()));

            for ( size_t i = 0; i < S.nr(); ++i ) indices.push_back( i );

            std::sort( indices.begin(), indices.end(),
                       [&]( size_t a, size_t b )
                       {
                         return S( a ) > S( b );
                       } );
            double sumSuares = std::accumulate( S.begin(), S.end(), double( 0 ),
                                                []( double acc, double s )
                                                {
                                                  return acc + s * s;
                                                } );
            double accVar = 0;
            size_t k = 0;
            for ( k = 0; k < indices.size() && accVar < exp; ++k )
            {
                double s = S( indices.at( k ));
                accVar += ( s * s ) / sumSuares;
            }
            dlib::matrix<double> Vk;
            Vk.set_size( S.nr(), k + 1 );
            for ( size_t i = 0; i <= k; ++i )
            {
                dlib::set_colm( Vk, i ) = dlib::colm( V, indices.at( i ));
            }
            return Vk;
        }
    }

    virtual std::vector<FeatureVector> _transformFeaturesLDA( std::vector<FeatureVector> &&features ) const
    {
        for ( auto &&f : features )
            f = _transformFeaturesLDA( std::move( f ));
        return features;
    }

    virtual std::vector<FeatureVector> _transformFeaturesSVD( std::vector<FeatureVector> &&features ) const
    {
        for ( auto &&f : features )
            f = _transformFeaturesSVD( std::move( f ));
        return features;
    }

    virtual FeatureVector _transformFeaturesLDA( FeatureVector &&f ) const
    {
        using namespace dlib_utilities;
        assert( _ldaConfiguration && _ldaData );
        if ( _ldaConfiguration && _ldaData )
        {
            auto &&Z = _ldaData->Z;
            auto &&M = _ldaData->M;
            auto x = vector_to_column_matrix_like( std::move( f ));
            column_matrix_like<double> fv( Z * x - M );
            return std::move( fv.steal_vector());
        } else return f;
    }

    virtual FeatureVector _transformFeaturesSVD( FeatureVector &&f ) const
    {
        using namespace dlib_utilities;
        assert( _svdConfiguration && _svdData );
        if ( _svdConfiguration && _svdData )
        {
            auto &&Vk = _svdData->Vk;
            auto &&M = _svdData->M;
            auto x = vector_to_row_matrix_like( std::move( f ));
            column_matrix_like<double> fv(( x - M ) * Vk );
            return std::move( fv.steal_vector());
        } else return f;
    }

    static std::vector<double> _centroid( const std::vector<std::vector<double >> &features )
    {
        const size_t ncol = features.front().size();
        assert( std::all_of( features.cbegin(), features.cend(),
                             [=]( auto &&v ) { return v.size() == ncol; } ));
        auto centroid = std::vector<double>( ncol, 0 );
        size_t n = 0;
        for ( auto &&fs : features )
        {
            for ( size_t col = 0; col < ncol; ++col )
            {
                if ( !std::isnan( fs[col] ))
                {
                    centroid[col] += fs[col];
                    ++n;
                }
            }
        }
        for ( auto &&m : centroid )
            m /= n;
        return centroid;
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
        centroid = _centroid( features );

        size_t n = 0;
        for ( auto &&fs : features )
        {
            for ( size_t col = 0; col < ncol; ++col )
            {
                if ( !std::isnan( fs[col] ))
                {
                    colMin[col] = std::min( colMin[col], fs[col] );
                    colMax[col] = std::max( colMax[col], fs[col] );
                    colMagnitude[col] += fs[col] * fs[col];
                    ++n;
                }
            }
        }

        for ( auto &&m : colMagnitude )
            m = std::sqrt( m );

        for ( auto &&fs : features )
        {
            for ( size_t col = 0; col < ncol; ++col )
            {
                if ( !std::isnan( fs[col] ))
                {
                    auto term = centroid[col] - fs[col];
                    colStandardDeviation[col] += term * term;
                }
            }
        }

        for ( auto &&s : colStandardDeviation )
            s = std::sqrt( s / n );
    }

    std::vector<std::vector<double >>
    _normalizeFeatures( std::vector<std::vector<double> > &&features ) const
    {
        for ( auto &&f : features )
            f = _normalizer.value()( std::move( f ));
        return features;
    }

    std::vector<std::vector<double >>
    _handleNans( std::vector<std::vector<double> > &&features ) const
    {
        for ( auto &&f : features )
            f = _nansHandler.value()( std::move( f ));
        return features;
    }

    std::vector<double> _normalizeFeatures( std::vector<double> &&features ) const
    {
        return _normalizer.value()( std::move( features ));
    }

    std::vector<double> _handleNans( std::vector<double> &&features ) const
    {
        return _nansHandler.value()( std::move( features ));
    }

    std::vector<double> _centerScaleColumns( std::vector<double> &&features ) const
    {
        auto &&centroid = _normData->centroid;
        auto &&colMax = _normData->colMax;
        auto &&colMin = _normData->colMin;

        assert( features.size() == colMin.size() && colMin.size() == colMax.size()
                && centroid.size() == colMax.size());
        for ( size_t col = 0; col < features.size(); ++col )
        {
            features[col] = ( features[col] - centroid[col] ) / ( colMax[col] - colMin[col] + eps );
        }
        return features;
    }

    std::vector<double> _minmaxScaleColumns( std::vector<double> &&features ) const
    {
        auto &&colMax = _normData->colMax;
        auto &&colMin = _normData->colMin;

        assert( features.size() == colMin.size() && colMin.size() == colMax.size());
        for ( size_t col = 0; col < features.size(); ++col )
        {
            features[col] = ( features[col] - colMin[col] ) / ( colMax[col] - colMin[col] + eps );
        }
        return features;
    }

    std::vector<double> _minmaxScaleRow( std::vector<double> &&features ) const
    {
        auto[minIt, maxIt] = std::minmax_element( features.cbegin(), features.cend());
        const double min = *minIt;
        const double max = *maxIt;
        for ( double &f : features )
            f = ( f - min ) / ( max - min );
        return features;
    }

    std::vector<double> _radiusNormalizeRow( std::vector<double> &&features ) const
    {
        auto[minIt, maxIt] = std::minmax_element( features.cbegin(), features.cend());
        const double min = *minIt;
        const double max = *maxIt;
        double magnitude = std::accumulate( features.cbegin(), features.cend(), double( 0 ),
                                            [](
                                                    double acc,
                                                    double val
                                            )
                                            {
                                              return acc + val * val;
                                            } );
        for ( double &f : features )
            f /= magnitude;
        return features;
    }

    std::vector<double> _radiusNormalizeColumns( std::vector<double> &&features ) const
    {
        auto &&colMagnitude = _normData->colMagnitude;

        assert( features.size() == colMagnitude.size());
        for ( size_t col = 0; col < features.size(); ++col )
            features[col] = features[col] / ( colMagnitude[col] + eps );
        return features;
    }

    std::vector<double> _standardNormalizeColumns( std::vector<double> &&features ) const
    {
        auto &&colStandardDeviation = _normData->colStandardDeviation;
        auto &&centroid = _normData->centroid;

        assert( features.size() == colStandardDeviation.size());
        for ( size_t col = 0; col < features.size(); ++col )
            features[col] = ( features[col] - centroid[col] ) / ( colStandardDeviation[col] + eps );
        return features;
    }

    std::vector<double> _handleNans(
            std::vector<double> &&featureVector,
            const std::vector<double> &compensation
    ) const
    {
        assert( compensation.size() == featureVector.size());
        for ( auto i = 0; i < featureVector.size(); ++i )
        {
            if ( std::isnan( featureVector[i] ))
                featureVector[i] = compensation[i];
        }
        return featureVector;
    }

private:
    std::optional<NormalizationData> _normData;
    std::optional<NormalizerFunction> _normalizer;
    std::optional<NaNsHandler> _nansHandler;
    std::optional<LDAConfiguration> _ldaConfiguration;
    std::optional<SVDConfiguration> _svdConfiguration;
    std::optional<LDAData> _ldaData;
    std::optional<SVDData> _svdData;
};

template < size_t States >
class MCStackedMLClassifier : public MCBasedMLModel<States>
{
public:
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using BackboneProfile = typename MCModel::BackboneProfile;
    using Generator = ModelGenerator<States>;

public:
    explicit MCStackedMLClassifier(
            Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : MCBasedMLModel<States>( generator, ldaConfig, pcaConfig ) {}

    void initWeakModels( SimilarityFunctor<Histogram> similarity )
    {
        using MacroSimilarityEnum = typename MicroSimilarityBasedClassifier<States>::MacroScoringEnum;
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


protected:
    FeatureVector _extractFeatures(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const override
    {
        FeatureVector f;
        for ( auto &[enumm, classifier] : this->_ensemble )
        {
            auto propensityPredictions =
                    classifier->scoredPredictions(
                            sequence,
                            backboneProfiles,
                            backgroundProfiles,
                            centralBackground ).toMap();

            for ( auto &[cluster, _] : backboneProfiles )
                f.push_back( propensityPredictions.at( cluster ));
        }
        f.push_back( sequence.length());
        return f;
    };
private:
    std::map<ClassificationEnum, std::unique_ptr<AbstractMCClassifier < States> >> _ensemble;
};

template < size_t States >
class KNNStackedMC : protected KNNModel<Euclidean>, public MCStackedMLClassifier<States>
{
    using KNN = KNNModel<Euclidean>;
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using Generator = ModelGenerator<States>;

public:
    explicit KNNStackedMC(
            size_t k,
            Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : KNN( k ), MCStackedMLClassifier<States>( generator, ldaConfig, pcaConfig ) {}

    virtual ~KNNStackedMC() = default;

    using MCBasedMLModel<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        KNN::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return KNN::predict( std::move( f ));
    }
};

template < size_t States >
class SVMStackedMC : protected SVMModel, public MCStackedMLClassifier<States>
{
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using Generator = ModelGenerator<States>;

public:
    explicit SVMStackedMC(
            SVMConfiguration configuration,
            const Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : MCStackedMLClassifier<States>( generator, ldaConfig, pcaConfig ),
              SVMModel( configuration ) {}

    virtual ~SVMStackedMC() = default;

    using MCBasedMLModel<States>::fit;
protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        SVMModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return SVMModel::predict( std::move( f ));
    }
};

template < size_t States >
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
            std::optional<SVDConfiguration> pcaConfig
    )
            : MCBasedMLModel<States>( generator, ldaConfig, pcaConfig ) {}

protected:
    FeatureVector _extractFeatures(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const override
    {
        assert( centralBackground ); // must exist.

        size_t nCols = std::accumulate( backboneProfiles.cbegin(), backboneProfiles.cend(), size_t( 0 ),
                                        [](
                                                size_t acc,
                                                auto &&kv
                                        )
                                        {
                                          return acc + kv.second->parametersCount();
                                        } );
        std::vector<double> flatFeatures;
        flatFeatures.reserve( nCols ); // TODO: handle exceptions.

        auto sample = this->_generator( sequence );

        centralBackground->centroids().get().forEach( [&](
                Order order,
                HistogramID id,
                auto &&backboneCentroid
        )
                                                      {

                                                        auto sampleCentroid = sample->centroid( order, id );
                                                        if ( sampleCentroid )
                                                        {
                                                            // auto standardDeviation = profile->standardDeviation( order, id );
                                                            flatFeatures.insert(
                                                                    flatFeatures.end(),
                                                                    sampleCentroid->get().cbegin(),
                                                                    sampleCentroid->get().cend());

                                                        } else
                                                        {
                                                            //auto standardDeviation = profile->standardDeviation( order, id );
                                                            flatFeatures.insert(
                                                                    flatFeatures.end(), backboneCentroid.cbegin(),
                                                                    backboneCentroid.cend());
                                                        }
                                                      } );

        return flatFeatures;
    }

private:
};

template < size_t States >
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
            std::optional<SVDConfiguration> pcaConfig
    )
            : SVMModel( configuration ),
              MCParametersBasedMLClassifier<States>( generator, ldaConfig, pcaConfig ) {}

    virtual ~SVMMCParameters() = default;

    using MCParametersBasedMLClassifier<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        SVMModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return SVMModel::predict( std::move( f ));
    }
};

template < size_t States >
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
            std::optional<SVDConfiguration> pcaConfig
    )
            : KNNModel( k ),
              MCParametersBasedMLClassifier<States>( generator, ldaConfig, pcaConfig ) {}

    virtual ~KNNMCParameters() = default;

    using MCParametersBasedMLClassifier<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        KNNModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return KNNModel::predict( std::move( f ));
    }
};

template < size_t States >
class RandomForestMCParameters : private RandomForestModel, public MCParametersBasedMLClassifier<States>
{
public:
    using FeatureVector = typename MCParametersBasedMLClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit RandomForestMCParameters(
            size_t nTrees,
            Generator generator,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : RandomForestModel( RandomForestConfiguration{nTrees} ),
              MCParametersBasedMLClassifier<States>( generator, ldaConfig, pcaConfig ) {}

    virtual ~RandomForestMCParameters() = default;

    using MCParametersBasedMLClassifier<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        RandomForestModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return RandomForestModel::predict( std::move( f ));
    }
};


template < size_t States >
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
            std::optional<SVDConfiguration> pcaConfig
    )
            : MCBasedMLModel<States>( generator, ldaConfig, pcaConfig ),
              _similarity( similarity ) {}

protected:
    template < typename MicroMeasurementsType, typename AlternativeMeasurementsType >
    static MicroMeasurements compensateMissingMeasurements(
            MicroMeasurementsType &&microMeasurements,
            AlternativeMeasurementsType &&alternativeMeasurements
    )
    {
        MicroMeasurements compensated = std::forward<MicroMeasurementsType>( microMeasurements );
        for ( auto &&[label, measurements] : microMeasurements )
        {
            for ( auto &&[order, isoMeasurements] : measurements )
            {
                auto &isoAlternativeMeasurements = alternativeMeasurements.at( order );
                for ( auto &&[id, measurement] : isoMeasurements )
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

    FeatureVector _extractFeatures(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const override
    {
        MicroMeasurements measurements;
        AlternativeMeasurements alternatives;
        auto closerThan = this->_similarity.closerThan;
        auto bestInfinity = this->_similarity.best;

        auto sample = this->_generator( sequence );
        std::map<std::string_view, std::vector<double >> similarities;
        for ( auto &[label, profile] : backboneProfiles )
        {
            auto &_measurements = measurements[label];
            auto &&backboneCentroids = profile->centroids().get();
            backboneCentroids.forEach( [&](
                    Order order,
                    HistogramID id,
                    auto &&backboneCentroid
            )
                                       {
                                         double &measurement = _measurements[order][id];
                                         double &furthest = alternatives[order].try_emplace( id,
                                                                                             bestInfinity ).first->second;

                                         auto sampleCentroid = sample->centroid( order, id );
                                         if ( sampleCentroid )
                                         {
                                             auto standardDeviation = profile->standardDeviation( order, id );
                                             measurement = this->_similarity( backboneCentroid, sampleCentroid->get(),
                                                                              standardDeviation->get());
                                         } else if ( centralBackground )
                                         {
                                             if ( auto center = centralBackground->centroid( order,
                                                                                             id ); center.has_value())
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
                                       [](
                                               size_t acc,
                                               auto &&kv
                                       )
                                       {
                                         return std::accumulate( kv.second.cbegin(), kv.second.cend(), acc,
                                                                 [](
                                                                         size_t acc,
                                                                         auto &&kv
                                                                 )
                                                                 {
                                                                   return acc + kv.second.size();
                                                                 } );
                                       } );

        std::vector<double> flatFeatures;
        flatFeatures.reserve( size );

        for ( auto &&[_, classMeasurements] : measurements )
            for ( auto &&[_, isoMeasurements]: classMeasurements )
                for ( auto &&[_, measurement]: isoMeasurements )
                    flatFeatures.push_back( measurement );

        return flatFeatures;
    }

private:
    const SimilarityFunctor<Histogram> _similarity;
};

template < size_t States >
class MCSinglePoleMicroSimilarityBasedMLClassifier : public MCBasedMLModel<States>
{
public:
    using FeatureVector = typename MCBasedMLModel<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using BackboneProfiles = typename MCModel::BackboneProfiles;
    using BackboneProfile = typename MCModel::BackboneProfile;
    using Generator = ModelGenerator<States>;
    using MicroMeasurements = std::unordered_map<Order, std::unordered_map<HistogramID, double >>;

public:
    explicit MCSinglePoleMicroSimilarityBasedMLClassifier(
            Generator generator,
            SimilarityFunctor<Histogram> similarity,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : MCBasedMLModel<States>( generator, ldaConfig, pcaConfig ),
              _similarity( similarity )
    {
        this->setNaNsHandling( NaNsHandlingEnum::Neutralize );
    }

protected:
    FeatureVector _extractFeatures(
            std::string_view sequence,
            const BackboneProfiles &backboneProfiles,
            const BackboneProfiles &backgroundProfiles,
            const BackboneProfile &centralBackground
    ) const override
    {
        MicroMeasurements measurements;
        MicroMeasurements alternatives;
        auto closerThan = this->_similarity.closerThan;
        auto bestInfinity = this->_similarity.best;

        auto sample = this->_generator( sequence );
        std::map<std::string_view, std::vector<double >> similarities;

        auto &&centralCentroids = centralBackground->centroids().get();
        centralCentroids.forEach( [&](
                Order order,
                HistogramID id,
                auto &&centralCentroid
        )
                                  {
                                    double &measurement = measurements[order][id];
                                    double &furthest = alternatives[order].try_emplace( id,
                                                                                        bestInfinity ).first->second;

                                    auto sampleCentroid = sample->centroid( order, id );
                                    if ( sampleCentroid )
                                    {
                                        auto standardDeviation = centralBackground->standardDeviation( order, id );
                                        measurement = this->_similarity( centralCentroid, sampleCentroid->get(),
                                                                         standardDeviation->get());
                                    } else
                                    {
                                        measurement = nan;
                                    }
                                  } );

        size_t size = centralBackground->histogramsCount();

        std::vector<double> flatFeatures;
        flatFeatures.reserve( size );

        for ( auto &&[_, isoMeasurements]: measurements )
            for ( auto &&[_, measurement]: isoMeasurements )
                flatFeatures.push_back( measurement );

        return flatFeatures;
    }

private:
    const SimilarityFunctor<Histogram> _similarity;
};

template < size_t States >
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
            std::optional<SVDConfiguration> pcaConfig
    )
            : SVMModel( configuration ),
              MCMicroSimilarityBasedMLClassifier<States>( generator, similarity, ldaConfig, pcaConfig ) {}

    virtual ~SVMCMMicroSimilarity() = default;

    using MCMicroSimilarityBasedMLClassifier<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        SVMModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return SVMModel::predict( std::move( f ));
    }
};

template < size_t States >
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
            std::optional<SVDConfiguration> pcaConfig
    )
            : KNNModel( k ),
              MCMicroSimilarityBasedMLClassifier<States>( generator, similarity, ldaConfig, pcaConfig ) {}

    virtual ~KNNMCMicroSimilarity() = default;

    using MCMicroSimilarityBasedMLClassifier<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        KNNModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return KNNModel::predict( std::move( f ));
    }
};


template < size_t States >
class RandomForestMicroSimilarity : private RandomForestModel, public MCMicroSimilarityBasedMLClassifier<States>
{
public:
    using FeatureVector = typename MCMicroSimilarityBasedMLClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit RandomForestMicroSimilarity(
            size_t nTrees,
            Generator generator,
            SimilarityFunctor<Histogram> similarity,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : RandomForestModel( RandomForestConfiguration{nTrees} ),
              MCMicroSimilarityBasedMLClassifier<States>( generator, similarity, ldaConfig, pcaConfig ) {}

    virtual ~RandomForestMicroSimilarity() = default;

    using MCMicroSimilarityBasedMLClassifier<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        RandomForestModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return RandomForestModel::predict( std::move( f ));
    }
};

template < size_t States >
class RandomForestSinglePoleMicroSimilarity
        : private RandomForestModel, public MCSinglePoleMicroSimilarityBasedMLClassifier<States>
{
public:
    using FeatureVector = typename MCSinglePoleMicroSimilarityBasedMLClassifier<States>::FeatureVector;
    using MCModel = AbstractMC<States>;
    using Histogram = typename MCModel::Histogram;
    using Generator = ModelGenerator<States>;

public:
    explicit RandomForestSinglePoleMicroSimilarity(
            size_t nTrees,
            Generator generator,
            SimilarityFunctor<Histogram> similarity,
            std::optional<LDAConfiguration> ldaConfig,
            std::optional<SVDConfiguration> pcaConfig
    )
            : RandomForestModel( RandomForestConfiguration{nTrees} ),
              MCSinglePoleMicroSimilarityBasedMLClassifier<States>( generator, similarity, ldaConfig, pcaConfig ) {}

    virtual ~RandomForestSinglePoleMicroSimilarity() = default;

    using MCSinglePoleMicroSimilarityBasedMLClassifier<States>::fit;

protected:
    void _fitML(
            std::vector<std::string_view> &&labels,
            std::vector<FeatureVector> &&f
    ) override
    {
        RandomForestModel::fit( std::move( labels ), std::move( f ));
    }

    ScoredLabels _predictML( FeatureVector &&f ) const override
    {
        return RandomForestModel::predict( std::move( f ));
    }
};
}
#endif //MARKOVIAN_FEATURES_MLCONFUSEDMC_HPP
