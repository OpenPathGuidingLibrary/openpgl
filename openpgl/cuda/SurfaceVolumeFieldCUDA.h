#pragma once

#include "field/ISurfaceVolumeField.h"
#include "directional/ISurfaceSamplingDistribution.h"
#include "directional/vmm/VMMPhaseFunctions.h"

#include "FieldCUDA.h"
#include "SampleStorageCUDA.h"

#include "../include/openpgl/gpu/Data.h"
#include "../include/openpgl/gpu/Data.h"

#include "data/Buffered.h"
#include "field/ISurfaceVolumeField.h"

namespace openpgl {
namespace OPENPGL_KERNEL_NS {

class SurfaceVolumeFieldCUDA {
public:

    SurfaceVolumeFieldCUDA(PGLFieldArguments args) {
        VMMSingleLobeHenyeyGreensteinOracle::init();

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

    void serializeIR(BufferedWriter& w) {
        m_surfaceField.serializeIR(w);
        m_volumeField.serializeIR(w);
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

        int numPhaseFunctionRepresentations = VMMSingleLobeHenyeyGreensteinOracle::representations.size();
        openpgl::gpu::VMMPhaseFunctionRepresentationData *phaseFunctionRepresentations = new openpgl::gpu::VMMPhaseFunctionRepresentationData[numPhaseFunctionRepresentations];
        for (int i = 0; i < numPhaseFunctionRepresentations; i++)
        {
            phaseFunctionRepresentations[i].K = VMMSingleLobeHenyeyGreensteinOracle::representations[i].K;
            phaseFunctionRepresentations[i].g = VMMSingleLobeHenyeyGreensteinOracle::representations[i].g;
            for (int j = 0; j < 4; j++)
                phaseFunctionRepresentations[i].weights[j] = VMMSingleLobeHenyeyGreensteinOracle::representations[i].weights[j];
            for (int j = 0; j < 4; j++)
                phaseFunctionRepresentations[i].meanCosines[j] = VMMSingleLobeHenyeyGreensteinOracle::representations[i].meanCosines[j];
            for (int j = 0; j < 4; j++)
                phaseFunctionRepresentations[i].kappas[j] = VMMSingleLobeHenyeyGreensteinOracle::representations[i].kappas[j];
        }
        fieldData->m_numPhaseFunctionRepresentations = numPhaseFunctionRepresentations;
        fieldData->m_phaseFunctionRepresentations = (void *)phaseFunctionRepresentations;
    };

    void releaseFieldData(openpgl::gpu::FieldData* fieldData) {
        TreeNode *deviceSurfNodes = (TreeNode *)fieldData->m_surfaceTreeLets;
        delete[] deviceSurfNodes;
        fieldData->m_surfaceTreeLets = nullptr;
        fieldData->m_numSurfaceTreeLets = 0;

        TreeNode *deviceVolumeNodes = (TreeNode *)fieldData->m_volumeTreeLets;
        delete[] deviceVolumeNodes;
        fieldData->m_volumeTreeLets = nullptr;
        fieldData->m_numVolumeTreeLets = 0;

        openpgl::gpu::FlatVMM<32> *outSurf = (openpgl::gpu::FlatVMM<32> *)fieldData->m_surfaceDistributions;
        delete[] outSurf;
        fieldData->m_surfaceDistributions = nullptr;
        // openpgl::gpu::OutgoingRadianceHistogramData* surfaceOutgoingRadianceHistogram = (openpgl::gpu::OutgoingRadianceHistogramData*)
        // fieldGPU->m_surfaceOutgoingRadianceHistogram; delete[] surfaceOutgoingRadianceHistogram; fieldGPU->m_surfaceOutgoingRadianceHistogram = nullptr;
        fieldData->m_numSurfaceDistributions = 0;

        openpgl::gpu::FlatVMM<32> *outVol = (openpgl::gpu::FlatVMM<32> *)fieldData->m_volumeDistributions;
        delete[] outVol;
        fieldData->m_volumeDistributions = nullptr;

        // openpgl::gpu::OutgoingRadianceHistogramData* volumeOutgoingRadianceHistogram = (openpgl::gpu::OutgoingRadianceHistogramData*)
        // fieldGPU->m_volumeOutgoingRadianceHistogram; delete[] volumeOutgoingRadianceHistogram; fieldGPU->m_volumeOutgoingRadianceHistogram = nullptr;
        fieldData->m_numVolumeDistributions = 0;

        openpgl::gpu::VMMPhaseFunctionRepresentationData *pfRep = (openpgl::gpu::VMMPhaseFunctionRepresentationData *)fieldData->m_phaseFunctionRepresentations;
        delete[] pfRep;
        fieldData->m_phaseFunctionRepresentations = nullptr;
    };

    void transferToCPUField(openpgl::ISurfaceVolumeField* fieldCPU) {
        std::vector<char> buf;
        BufferedWriter w(buf);
        serializeIR(w);
        BufferedReader r(buf);
        fieldCPU->deserializeIR(r);
    }

    size_t m_iteration{0};
    size_t m_totalSPP{0};

    FieldCUDA m_surfaceField;
    FieldCUDA m_volumeField;
};

}
}