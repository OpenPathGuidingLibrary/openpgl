#pragma once

#include "field/ISurfaceVolumeField.h"
#include "directional/ISurfaceSamplingDistribution.h"

#include "FieldCUDA.h"
#include "SampleStorageCUDA.h"

#include "../include/openpgl/gpu/Data.h"

namespace openpgl {
namespace OPENPGL_KERNEL_NS {

class SurfaceVolumeFieldCUDA {
public:

    SurfaceVolumeFieldCUDA(PGLFieldArguments args) {
        if (args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && args.directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM)
        {
            //typename GuidingField::Settings gFieldSettings;
            //gFieldSettings.settings.decayOnSpatialSplit = 0.25f;
            //gFieldSettings.settings.deterministic = args.deterministic;
            //gFieldSettings.debugSettings.fitRegions = args.debugArguments.fitRegions;
            //
            //PGLKDTreeArguments *sargs = (PGLKDTreeArguments *)args.spatialSturctureArguments;
            //gFieldSettings.settings.useStochasticNNLookUp = sargs->knnLookup;
            //gFieldSettings.settings.useISNNLookUp = sargs->isKnnLookup;
            //gFieldSettings.settings.spatialSubdivBuilderSettings.minSamples = sargs->minSamples;
            //gFieldSettings.settings.spatialSubdivBuilderSettings.maxSamples = sargs->maxSamples;
            //gFieldSettings.settings.spatialSubdivBuilderSettings.maxDepth = sargs->maxDepth;
            //delete sargs;

            PGLVMMFactoryArguments *dargs = (PGLVMMFactoryArguments *)args.directionalDistributionArguments;
            Factory::Configuration dcfg;
            dcfg.weightedEMCfg.initK = dargs->initK;
            dcfg.weightedEMCfg.initKappa = dargs->initKappa;
            dcfg.weightedEMCfg.maxK = dargs->maxK;
            dcfg.weightedEMCfg.maxEMIterrations = dargs->maxEMIterrations;

            dcfg.weightedEMCfg.maxKappa = dargs->maxKappa;
            dcfg.weightedEMCfg.maxMeanCosine = KappaToMeanCosine<float>(dcfg.weightedEMCfg.maxKappa);
            dcfg.weightedEMCfg.convergenceThreshold = dargs->convergenceThreshold;
            dcfg.weightedEMCfg.weightPrior = dargs->weightPrior;
            dcfg.weightedEMCfg.meanCosinePriorStrength = dargs->meanCosinePriorStrength;
            dcfg.weightedEMCfg.meanCosinePrior = dargs->meanCosinePrior;

            dcfg.splittingThreshold = dargs->splittingThreshold;
            dcfg.mergingThreshold = dargs->mergingThreshold;

            dcfg.partialReFit = dargs->partialReFit;
            dcfg.maxSplitItr = dargs->maxSplitItr;

            dcfg.useSplitAndMerge = dargs->useSplitAndMerge;
            dcfg.minSamplesForSplitting = dargs->minSamplesForSplitting;
            dcfg.minSamplesForPartialRefitting = dargs->minSamplesForPartialRefitting;
            dcfg.minSamplesForMerging = dargs->minSamplesForMerging;
            delete dargs;

            m_surfaceField.dcfg = dcfg;
            m_volumeField.dcfg = dcfg;
        }
        else
        {
            throw std::runtime_error("error: unrecognized field type");
        }
    }

    void update(SampleStorageCUDA &sampleStorage) {
        sampleStorage.alloc = sampleStorage.allocDevice[0];
        m_surfaceField.Update(
            std::min(sampleStorage.alloc.sizeSurface, sampleStorage.alloc.capacity),
            sampleStorage.samplesSurface
        );
        m_volumeField.Update(
            std::min(sampleStorage.alloc.sizeVolume, sampleStorage.alloc.capacity),
            sampleStorage.samplesVolume
        );
        m_iteration++;
    }

    void dump(const std::string &dumpFileName) {
        m_volumeField.dump(dumpFileName + ".volume.dump");
        m_surfaceField.dump(dumpFileName + ".surface.dump");
    }

    void fillFieldData(openpgl::gpu::FieldData* fieldData) {
        fieldData->m_ready = m_iteration > 0;
        m_surfaceField.fillFieldData(
            fieldData->m_numSurfaceTreeLets, &fieldData->m_surfaceTreeLets,
            fieldData->m_numSurfaceDistributions, &fieldData->m_surfaceDistributions
        );
        m_volumeField.fillFieldData(
            fieldData->m_numVolumeTreeLets, &fieldData->m_volumeTreeLets,
            fieldData->m_numVolumeDistributions, &fieldData->m_volumeDistributions
        );
    };

    size_t m_iteration{0};
    size_t m_totalSPP{0};

    FieldCUDA m_surfaceField;
    FieldCUDA m_volumeField;
};

}
}