//Copyright (C) 2013 Sergio E. Gonzalez and Emiliano D. González
//Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires, Buenos Aires, Argentina.
// 
//C/C++, Java and XML/YML code for EKF SLAM from a monocular sequence.
//
//This file is part of OpenEKFMonoSLAM.
//
//OpenEKFMonoSLAM is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//OpenEKFMonoSLAM is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with OpenEKFMonoSLAM.  If not, see <http:www.gnu.org/licenses/>.
//
//If you use this code for academic work, please reference:
//GONZALEZ, S.; E. GONZÁLEZ; M. NITSCHE; P. DE CRISTÓFORIS. "Odometría Visual para Robots Móviles Utilizando Smartphones como Unidad de Sensado y Procesamiento". En: Jornadas Argentinas de Robótica, 8as : 2014 : Ciudad Autónoma de Buenos Aires. Actas : VIII Jornadas Argentinas de Robótica . (8 : Ciudad Autónoma de Buenos Aires, 12-14 de noviembre 2014).
//
//Authors:    Sergio E. Gonzalez - segonzalez@dc.uba.ar
//            Emiliano D. González - edgonzalez@dc.uba.ar
//
//Departamento de Computación
//Facultad de Ciencias Exactas y Naturales
//Universidad de Buenos, Buenos Aires, Argentina
//Date: June 2013

#include "../Core/EKFMath.h"
#include "../Configuration/ConfigurationManager.h"
#include "../Core/Timer.h"

#include "CommonFunctions.h"

#include "State.h"
#include "MapFeature.h"
#include "ImageFeaturePrediction.h"
#include "ImageFeatureMeasurement.h"

#include "StateAndCovariancePrediction.h"
#include "MeasurementPrediction.h"
#include "Matching.h"
#include "Update.h"

#include "AddMapFeature.h"
#include "DetectNewImageFeatures.h"
#include "MapManagement.h"

#include "1PointRansac.h"

#include "EKF.h"

#include "../Gui/Draw.h"

#if defined(DEBUG_SHOW_IMAGES) || defined(DEBUG_SHOW_RANSAC_INFO)
#include <opencv2/highgui/highgui.hpp>
#endif


// --------------------------------------------------------------------------------------------------------------------
// Rescata los ouliers que se consideran buenos
// IMPORTANTE: Asume que outlierMatches, outlierMatchFeaturePrediction y (por fuera outlierMatchFeaturePredictionJacobians)
// tienen la misma cantidad de elementos y se corresponden en cada posicion los unos con los otros.
// --------------------------------------------------------------------------------------------------------------------
void rescueOutliers(const VectorFeatureMatch &outlierMatches,
                    const VectorImageFeaturePrediction &outlierMatchFeaturePrediction,
                    const VectorMatd &outlierMatchFeaturePredictionJacobians,
                    VectorFeatureMatch &rescuedMatches,
                    VectorImageFeaturePrediction &rescuedPredictions,
                    VectorMatd &rescuedJacobians)
{
    ExtendedKalmanFilterParameters *ekfParams = ConfigurationManager::getInstance().ekfParams;
    std::vector<int> rescuedMatchesIndexes;

    size_t outlierMatchesSize = outlierMatches.size();
    for (uint i = 0; i < outlierMatchesSize; ++i)
    {
        FeatureMatch *match = outlierMatches[i];
        ImageFeaturePrediction *prediction = outlierMatchFeaturePrediction[i];

        Matd dist = (Matd(2, 1) << match->imagePos[0] - prediction->imagePos[0],
                                   match->imagePos[1] - prediction->imagePos[1]);

        // Se verifica que sea un buen match
        // http://es.wikipedia.org/wiki/Propagaci%C3%B3n_de_errores
        // Se propaga el error desde la funcion que calcula la prediccion a la funcion
        // que calcula la distancia entre la prediccion y el match
        // entonces, el calculo de esa distancia, tiene que tener un error menor a ransacChi2Threshold
        // Osea que es una forma de decir que la confianza que se tiene sobre dicha distancia es muy grande
        // y queda en funcion de la matriz de covarianza de la prediccion
        if ( Matd( dist.t() * prediction->covarianceMatrix.inv() * dist)[0][0] < ekfParams->ransacChi2Threshold )
        {
            rescuedMatchesIndexes.push_back(i);
        }
    }

    // Se agregan los resultados para los indices rescatados.
    rescuedMatches.clear();
    rescuedPredictions.clear();
    rescuedJacobians.clear();

    // Se reserva lugar para que los push_back sean en O(1)
    rescuedMatches.reserve(rescuedMatchesIndexes.size());
    rescuedPredictions.reserve(rescuedMatchesIndexes.size());
    rescuedJacobians.reserve(rescuedMatchesIndexes.size());

    for (uint i = 0; i < rescuedMatchesIndexes.size(); ++i)
    {
        int index = rescuedMatchesIndexes[i];

        rescuedMatches.push_back(outlierMatches[index]);
        rescuedPredictions.push_back(outlierMatchFeaturePrediction[index]);
        rescuedJacobians.push_back(outlierMatchFeaturePredictionJacobians[index]);
    }

}


//-------------------------------------------------------------------------------------------------------------

EKF::EKF(const char *configurationFileName, const char *outputPath) : _ekfSteps(0), _strOutputPath(outputPath)
{
    ConfigurationManager &configManager = ConfigurationManager::getInstance();
    configManager.loadConfigurationFromFile(configurationFileName);
}

//-------------------------------------------------------------------------------------------------------------

EKF::~EKF()
{
    if (_outputFileStorage.isOpened())
    {
        _outputFileStorage.release();
    }

    if (_logFile.is_open())
    {
        _logFile.close();
    }

#ifndef ANDROID
    if (_outputVideoWriter.isOpened())
    {
        _outputVideoWriter.release();
    }
#endif
}

//-------------------------------------------------------------------------------------------------------------

void EKF::init(const cv::Mat &image)
{
    ExtendedKalmanFilterParameters *ekfParams = ConfigurationManager::getInstance().ekfParams;

    initState(state);

    state.mapFeatures.reserve(ekfParams->reserveFeaturesDepth);
    state.mapFeaturesDepth.reserve(ekfParams->reserveFeaturesDepth);
    state.mapFeaturesInvDepth.reserve(ekfParams->reserveFeaturesInvDepth);

    initCovariance(stateCovarianceMatrix);

    // Detectar features en la imagen
    VectorFeatureMatch noMatches; // Al principio no tiene nada ya que no hay matches
    VectorImageFeaturePrediction noPredictions;
    VectorImageFeatureMeasurement newFeatureMeasurements;
    detectNewImageFeatures(image, noPredictions, ekfParams->minMatchesPerImage, newFeatureMeasurements);

    // Agregar los features nuevos al estado
    addFeaturesToStateAndCovariance(newFeatureMeasurements, state, stateCovarianceMatrix);

    size_t newFeatureMeasurementsSize = newFeatureMeasurements.size();
    for (uint i = 0; i < newFeatureMeasurementsSize; ++i)
    {
        delete newFeatureMeasurements[i];
    }
}

// ------------------------------------------------------------------------------------------------------------
// Se hace un paso del EKF con prediccion, matching, ransac y update.
// ------------------------------------------------------------------------------------------------------------
void EKF::step(const cv::Mat &image)
{
    _ekfSteps++;

    //-------------------------------------------------------------------------------------------------------------
    // Prediccion

    Timer timer;

    VectorImageFeaturePrediction predictedDistortedFeatures;
    VectorMatd predictedFeatureJacobians;

    stateAndCovariancePrediction(state, stateCovarianceMatrix);

    // se le pasa mapFeatureIndexes vacio para que sepa que la prediccion es para todos los features en el mapa
    std::vector<int> mapFeatureIndexes;
    VectorMapFeature unseenFeatures;
    predictCameraMeasurements( state,
                               stateCovarianceMatrix,
                               state.mapFeatures,
                               mapFeatureIndexes,
                               predictedDistortedFeatures,
                               predictedFeatureJacobians,
                               unseenFeatures );
    //-------------------------------------------------------------------------------------------------------------
    // Medicion y Matching
    VectorFeatureMatch matches;

    matchPredictedFeatures(image, state.mapFeatures, predictedDistortedFeatures, matches);

    //-------------------------------------------------------------------------------------------------------------
    // Update con Low Innovation

    VectorImageFeaturePrediction predictedMatchedFeatures;
    VectorMatd predictedMatchedJacobians;
    size_t predictedDistortedFeaturesSize = predictedDistortedFeatures.size();

    predictedMatchedFeatures.reserve(matches.size());
    predictedMatchedJacobians.reserve(matches.size());

    // De los features predichos, se dejan aquellos que matchearon, y en el mismo orden que "matches"
    // como los predichos estan en el mismo orden que sus covarianzas, también se dejan las covarianzas ordenadas
    for (uint i = 0; i < matches.size(); ++i)
    {
        int featureIndex = matches[i]->featureIndex;
        bool wasFound = false;
        int predictedFeatureIndex = 0;

        while (!wasFound && predictedFeatureIndex < predictedDistortedFeaturesSize)
        {
            ImageFeaturePrediction *featurePrediction = predictedDistortedFeatures[predictedFeatureIndex];

            if (featureIndex == featurePrediction->featureIndex)
            {
                // Se agrega la prediccion
                predictedMatchedFeatures.push_back(featurePrediction);

                // Se agregan los jacobianos de dicha prediccion
                predictedMatchedJacobians.push_back(predictedFeatureJacobians[predictedFeatureIndex]);

                wasFound = true;
            }

            predictedFeatureIndex++;
        }

    }

    VectorFeatureMatch inlierMatches;
    VectorImageFeaturePrediction inlierPredictions;
    VectorMatd inlierJacobians;

    // Los outliers se necesitan para identificar que features hay que sacar luego de tantas iteraciones sin matchear y para hacer el rescate
    VectorFeatureMatch outlierMatches;

    // se buscan matches inliers, que formen un consenso para hacer una buena estimacion
    ransac(state, stateCovarianceMatrix, predictedMatchedFeatures, predictedMatchedJacobians, matches,
           inlierMatches, inlierPredictions, inlierJacobians, outlierMatches);

    // Se actualiza el estado tratando de recuperar ouliers
    update(state, stateCovarianceMatrix, inlierMatches, inlierPredictions, inlierJacobians);

    //-------------------------------------------------------------------------------------------------------------
    // Rescate de outliers

    VectorFeatureMatch rescuedMatches;
    VectorImageFeaturePrediction rescuedPredictions;
    VectorMatd rescuedJacobians;

    // Se hace una prediccion para buscar otros puntos que sean buenos
    VectorImageFeaturePrediction outlierMatchFeaturePrediction;
    VectorMatd outlierMatchFeaturePredictionJacobians;

    //Se buscan los features del mapa que se corresponden con los oulierMatches
    VectorMapFeature outlierFeatures;
    size_t outlierMatchesSize = outlierMatches.size();
    std::vector<int> outlierFeatureIndexes;

    outlierFeatures.reserve(outlierMatchesSize);
    outlierFeatureIndexes.reserve(outlierMatchesSize);

    for (uint i = 0; i < outlierMatchesSize; ++i)
    {
        int featureIndex = outlierMatches[i]->featureIndex;

        outlierFeatures.push_back(state.mapFeatures[featureIndex]);
        outlierFeatureIndexes.push_back(featureIndex);
    }

    VectorMapFeature unseenOutliers;
    predictCameraMeasurements( state, stateCovarianceMatrix,
                               outlierFeatures, outlierFeatureIndexes,
                               outlierMatchFeaturePrediction, outlierMatchFeaturePredictionJacobians,
                               unseenOutliers );

    // Tenemos que filtrar los matches tal que su prediccion cayo fuera de la imagen
    // Para eso aprovechamos el orden relativo de las predicciones, que es el mismo
    // que el orden relativo de los features que se pasaron para predecir
    size_t outlierMatchFeaturePredictionSize = outlierMatchFeaturePrediction.size();

    if (0 < outlierMatchFeaturePredictionSize && outlierMatchFeaturePredictionSize < outlierMatchesSize)
    {
        VectorFeatureMatch outlierMatchesPredictedInsideFrame;
        outlierMatchesPredictedInsideFrame.reserve(outlierMatchFeaturePredictionSize);

        uint j = 0;
        for (uint i = 0; i < outlierMatchesSize && j < outlierMatchFeaturePredictionSize; ++i)
        {
            if (outlierMatches[i]->featureIndex == outlierMatchFeaturePrediction[j]->featureIndex)
            {
                j++;
                outlierMatchesPredictedInsideFrame.push_back(outlierMatches[i]);
            }
        }

        outlierMatches = outlierMatchesPredictedInsideFrame;
    }

    if (outlierMatches.size())
    {
        // Se intentan rescatar posibles matches que eran outliers antes del primer update
        rescueOutliers( outlierMatches, outlierMatchFeaturePrediction, outlierMatchFeaturePredictionJacobians,
                        rescuedMatches, rescuedPredictions, rescuedJacobians );
    }
  
    //-------------------------------------------------------------------------------------------------------------
    // Update con High Innovation

    // Si se rescataron outliers entonces actualizar el ekf
    size_t rescuedMatchesSize = rescuedMatches.size();
    if (rescuedMatchesSize)
    {
        update(state, stateCovarianceMatrix, rescuedMatches, rescuedPredictions, rescuedJacobians);
    }

    //-------------------------------------------------------------------------------------------------------------
    // Map management: se agregan y se eliminan features del mapa

    // Juntamos los inlierMatches con los outliers rescatados y tambien sus predicciones
    inlierMatches.reserve(inlierMatches.size() + rescuedMatchesSize);

    for (int i = 0; i < rescuedMatchesSize; ++i)
    {
        inlierMatches.push_back(rescuedMatches[i]);
        inlierPredictions.push_back(rescuedPredictions[i]);
    }

    updateMapFeatures(predictedDistortedFeatures, inlierMatches, state);

    ExtendedKalmanFilterParameters *ekfParams = ConfigurationManager::getInstance().ekfParams;
    if (ekfParams->mapManagementFrequency > 0 && _ekfSteps % ekfParams->mapManagementFrequency == 0)
    {
        int newFeaturesNeededCount = ekfParams->minMatchesPerImage - static_cast<int>(inlierMatches.size());

        removeBadMapFeatures(state, stateCovarianceMatrix);

        // Elimino los features que no se ven en el frame actual segun el criterio configurado
        if ( newFeaturesNeededCount > 0 && ( ekfParams->alwaysRemoveUnseenMapFeatures ||
             (ekfParams->maxMapFeaturesCount > 0 && state.mapFeatures.size() + newFeaturesNeededCount > ekfParams->maxMapFeaturesCount) ||
             (ekfParams->maxMapSize > 0 && stateCovarianceMatrix.rows + newFeaturesNeededCount*6 > ekfParams->maxMapSize) ) )
        {
            removeFeaturesFromStateAndCovariance(unseenFeatures, state, stateCovarianceMatrix);
            //resetEKFMap(state, stateCovarianceMatrix);
            //newFeaturesNeededCount = ekfParams->minMatchesPerImage;
        }

        convertMapFeaturesInverseDepthToDepth(state, stateCovarianceMatrix);

        // Agregamos nuevos features en caso de ser necesario
        if (newFeaturesNeededCount > 0)
        {
            // Detectar features en la imagen
            VectorImageFeatureMeasurement newFeatureMeasurements;
            detectNewImageFeatures(image, predictedDistortedFeatures, newFeaturesNeededCount, newFeatureMeasurements);

            // Agrego los nuevos features detectados al mapa
            addFeaturesToStateAndCovariance(newFeatureMeasurements, state, stateCovarianceMatrix);

            size_t newFeatureMeasurementsSize = newFeatureMeasurements.size();
            for (uint i = 0; i < newFeatureMeasurementsSize; ++i)
            {
                delete newFeatureMeasurements[i];
            }
        }
    }

    //-------------------------------------------------------------------------------------------------------------
    // Se libera la memoria auxiliar utilizada
    size_t predictedFeatureJacobiansSize = predictedFeatureJacobians.size();
    size_t matchesSize = matches.size();
    size_t outlierMatchFeaturePredictionJacobiansSize = outlierMatchFeaturePredictionJacobians.size();

    for (uint i = 0; i < predictedDistortedFeaturesSize; ++i)
    {
        delete predictedDistortedFeatures[i];
    }

    for (uint i = 0; i < predictedFeatureJacobiansSize; ++i)
    {
        delete predictedFeatureJacobians[i];
    }

    for (uint i = 0; i < matchesSize; ++i)
    {
        delete matches[i];
    }

    for (uint i = 0; i < outlierMatchFeaturePredictionSize; ++i)
    {
        delete outlierMatchFeaturePrediction[i];
    }

    for (uint i = 0; i < outlierMatchFeaturePredictionJacobiansSize; ++i)
    {
        delete outlierMatchFeaturePredictionJacobians[i];
    }
}
