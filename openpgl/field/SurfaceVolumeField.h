// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../directional/vmm/VMMPhaseFunctions.h"
#include "../include/openpgl/gpu/Data.h"
#include "../include/openpgl/gpu/Device.h"
#include "Field.h"
#include "FieldStatistics.h"
#include "ISurfaceVolumeField.h"

#define FIELD_FILE_HEADER_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_FIELD"

namespace openpgl
{
namespace OPENPGL_KERNEL_NS
{
template <int maxComponents>
struct FlatVMM
{
    float _weights[maxComponents];
    float _kappas[maxComponents];
    float _meanDirections[maxComponents][3];
    float _distances[maxComponents];
    float _pivotPosition[3];
    int _numComponents{maxComponents};
};

template <int Vecsize, class TDirectionalDistributionFactory, template <typename, typename, typename> class TSpatialStructureBuilder, typename TSurfaceSamplingDistribution,
          typename TVolumeSamplingDistribution>
struct SurfaceVolumeField : public ISurfaceVolumeField
{
   private:
    using FieldType = Field<Vecsize, TDirectionalDistributionFactory, TSpatialStructureBuilder>;
    using SampleContainer = SampleDataStorage::SampleContainer;

   public:
    using Settings = typename FieldType::Settings;
    using RegionType = typename FieldType::RegionType;
    using DirectionalDistribution = typename FieldType::DirectionalDistribution;

   public:
    SurfaceVolumeField() = default;

    SurfaceVolumeField(const Settings &settings) : m_surfaceField(settings), m_volumeField(settings)
    {
        m_surfaceField.setIsSurface(true);
        m_volumeField.setIsSurface(false);
    }

    ~SurfaceVolumeField() override {}

    ISurfaceSamplingDistribution *newSurfaceSamplingDistribution() const override
    {
        return new TSurfaceSamplingDistribution();
    }

    bool initSurfaceSamplingDistribution(ISurfaceSamplingDistribution *surfaceSamplingDistribution, const Point3 &position, float *sample1D) const override
    {
        TSurfaceSamplingDistribution *_surfaceSamplingDistribution = (TSurfaceSamplingDistribution *)surfaceSamplingDistribution;
        uint32_t id = -1;
        const RegionType *region = m_surfaceField.getRegion(position, sample1D, id);
        if (!region || !region->valid || !region->initialized)
        {
            return false;
        }
        const DirectionalDistribution *distribution = &region->distribution;
        _surfaceSamplingDistribution->init(distribution, position);
        _surfaceSamplingDistribution->setId(id);
        _surfaceSamplingDistribution->setRegion(region);
        return true;
    }

    IVolumeSamplingDistribution *newVolumeSamplingDistribution() const override
    {
        return new TVolumeSamplingDistribution();
    }

    bool initVolumeSamplingDistribution(IVolumeSamplingDistribution *volumeSamplingDistribution, const Point3 &position, float *sample1D) const override
    {
        TVolumeSamplingDistribution *_volumeSamplingDistribution = (TVolumeSamplingDistribution *)volumeSamplingDistribution;
        uint32_t id = -1;
        const RegionType *region = m_volumeField.getRegion(position, sample1D, id);
        if (!region || !region->valid || !region->initialized)
        {
            return false;
        }
        const DirectionalDistribution *distribution = region->getDistribution(position);
        _volumeSamplingDistribution->init(distribution, position);
        _volumeSamplingDistribution->setId(id);
        _volumeSamplingDistribution->setRegion(region);
        return true;
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds) override
    {
        openpgl::BBox scaledSceneBounds = sceneBounds;
        scaledSceneBounds.enlarge_by(1.01f);
        m_surfaceField.setSceneBounds(scaledSceneBounds);
        m_volumeField.setSceneBounds(scaledSceneBounds);
    }

    openpgl::BBox getSceneBounds() const override
    {
        openpgl::BBox sceneBounds = m_surfaceField.getSceneBounds();
        sceneBounds.extend(m_volumeField.getSceneBounds());
        return sceneBounds;
    }

    void dumpField(const std::string dumpFileName) const override
    {
        m_volumeField.dump(dumpFileName + ".volume.dump");
        m_surfaceField.dump(dumpFileName + ".surface.dump");
    }

    void updateField(SampleContainer &samplesSurface, SampleContainer &samplesVolume) override
    {
#if TBB_INTERFACE_VERSION < 12010
        // we need to initialize the task_scheduler in the context to avoid
        // asyncronous deconsrution of the implicit initialized tbb::arenas and tbb::streams
        tbb::task_scheduler_init anonymous;
#endif
        if (samplesSurface.samples.size() > 0)
        {
            if (!m_surfaceField.isInitialized())
            {
                m_surfaceField.buildField(samplesSurface);
            }
            else
            {
                m_surfaceField.updateField(samplesSurface);
            }
        }
        if (samplesVolume.samples.size() > 0)
        {
            if (!m_volumeField.isInitialized())
            {
                m_volumeField.buildField(samplesVolume);
            }
            else
            {
                m_volumeField.updateField(samplesVolume);
            }
        }
        m_iteration++;
    }

    void updateFieldSurface(SampleContainer &samplesSurface) override
    {
        if (samplesSurface.samples.size() > 0)
        {
            if (!m_surfaceField.isInitialized())
            {
                m_surfaceField.buildField(samplesSurface);
            }
            else
            {
                m_surfaceField.updateField(samplesSurface);
            }
        }
        m_iteration++;
    }

    void updateFieldVolume(SampleContainer &samplesVolume) override
    {
        if (samplesVolume.samples.size() > 0)
        {
            if (!m_volumeField.isInitialized())
            {
                m_volumeField.buildField(samplesVolume);
            }
            else
            {
                m_volumeField.updateField(samplesVolume);
            }
        }
        m_iteration++;
    }

    void resetField() override
    {
        m_iteration = 0;
        m_totalSPP = 0;
        m_surfaceField.resetField();
        m_volumeField.resetField();
    }

    PGL_SPATIAL_STRUCTURE_TYPE getSpatialStructureType() const override
    {
        return FieldType::SpatialStructureBuilder::SPATIAL_STRUCTURE_TYPE;
    }

    PGL_DIRECTIONAL_DISTRIBUTION_TYPE getDirectionalDistributionType() const override
    {
        return FieldType::DirectionalDistributionFactory::DIRECTIONAL_DISTRIBUTION_TYPE;
    }

    size_t getIteration() const override
    {
        return m_iteration;
    }

    void serialize(std::ostream &os) const override
    {
        os.write(reinterpret_cast<const char *>(&m_iteration), sizeof(m_iteration));
        os.write(reinterpret_cast<const char *>(&m_totalSPP), sizeof(m_totalSPP));
        m_surfaceField.serialize(os);
        m_volumeField.serialize(os);
    }

    void deserialize(std::istream &is) override
    {
        is.read(reinterpret_cast<char *>(&m_iteration), sizeof(m_iteration));
        is.read(reinterpret_cast<char *>(&m_totalSPP), sizeof(m_totalSPP));
        m_surfaceField.deserialize(is);
        m_volumeField.deserialize(is);
    }

    virtual bool validate(const bool checkSurface, const bool checkVolume) const override
    {
        bool valid = true;
        if (m_surfaceField.isInitialized())
            valid = valid & m_surfaceField.isValid();
        if (m_volumeField.isInitialized())
            valid = valid & m_volumeField.isValid();
        return valid;
    }

    void storeToFile(const std::string fieldFileName) const override
    {
        std::filebuf fb;
        fb.open(fieldFileName, std::ios::out | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file!");
        std::ostream os(&fb);

        os.write(FIELD_FILE_HEADER_STRING, strlen(FIELD_FILE_HEADER_STRING) + 1);

        auto spatialStructureType = FieldType::SpatialStructureBuilder::SPATIAL_STRUCTURE_TYPE;
        os.write(reinterpret_cast<const char *>(&spatialStructureType), sizeof(spatialStructureType));
        auto directionalDistributionType = FieldType::DirectionalDistributionFactory::DIRECTIONAL_DISTRIBUTION_TYPE;
        os.write(reinterpret_cast<const char *>(&directionalDistributionType), sizeof(directionalDistributionType));

        serialize(os);

        os.flush();
        fb.close();
    }

    virtual bool operator==(const ISurfaceVolumeField *b) const override
    {
        bool equal = true;
        const SurfaceVolumeField *fieldB = dynamic_cast<const SurfaceVolumeField *>(b);
        if (!fieldB || m_iteration != fieldB->m_iteration || m_totalSPP != fieldB->m_totalSPP || !m_surfaceField.operator==(fieldB->m_surfaceField) ||
            !m_volumeField.operator==(fieldB->m_volumeField))
        {
            equal = false;
        }
        return equal;
    }

    FieldStatistics *getSurfaceStatistics() const override
    {
        FieldStatistics *stats = m_surfaceField.getStatistics();
        return stats;
    }

    FieldStatistics *getVolumeStatistics() const override
    {
        FieldStatistics *stats = m_volumeField.getStatistics();
        return stats;
    }
    /*
        virtual int GetNumNodes(bool isSurface = true) const override
        {
            if(isSurface)
            {
                return m_surfaceField.m_spatialSubdiv.m_numTreeLets;
            }
            else
            {
                return m_volumeField.m_spatialSubdiv.m_numTreeLets;
            }

        }

        virtual void *GetNodes(bool isSurface = true) const override
        {
            if(isSurface)
            {
                return m_surfaceField.m_spatialSubdiv.m_treeLets;
            }
            else
            {
                return m_volumeField.m_spatialSubdiv.m_treeLets;
            }
        }

        virtual int GetNumDistributions(bool isSurface = true) const override
        {
            if(isSurface)
            {
                return m_surfaceField.m_regionStorageContainer.size();
            }
            else
            {
                return m_volumeField.m_regionStorageContainer.size();
            }
        }

        virtual void CopyDistributionsTo(void *o_distrib, bool isSurface = true) const override
        {
            if(isSurface)
            {
                openpgl::gpu::FlatVMM<32> *out = reinterpret_cast<openpgl::gpu::FlatVMM<32>*>(o_distrib);
                for (int i = 0; i < m_surfaceField.m_regionStorageContainer.size(); i++)
                {
                    auto & dist = m_surfaceField.m_regionStorageContainer[i].first.distribution;
                    for (int k = 0; k < dist._numComponents; k++) {
                        const div_t tmp = div(k, static_cast<int>(Vecsize));
                        out[i]._weights[k] = dist._weights[tmp.quot][tmp.rem];
                        out[i]._kappas[k] = dist._kappas[tmp.quot][tmp.rem];
                        out[i]._meanDirections[k][0] = dist._meanDirections[tmp.quot].x[tmp.rem];
                        out[i]._meanDirections[k][1] = dist._meanDirections[tmp.quot].y[tmp.rem];
                        out[i]._meanDirections[k][2] = dist._meanDirections[tmp.quot].z[tmp.rem];
                        out[i]._distances[k] = dist._distances[tmp.quot][tmp.rem];
                    }
                    out[i]._pivotPosition[0] = {dist._pivotPosition.x};
                    out[i]._pivotPosition[1] = {dist._pivotPosition.y};
                    out[i]._pivotPosition[2] = {dist._pivotPosition.z};
                    out[i]._numComponents = dist._numComponents;
                }
            }
            else
            {
                openpgl::gpu::FlatVMM<32> *out = reinterpret_cast<openpgl::gpu::FlatVMM<32>*>(o_distrib);
                for (int i = 0; i < m_volumeField.m_regionStorageContainer.size(); i++)
                {
                    auto & dist = m_volumeField.m_regionStorageContainer[i].first.distribution;
                    for (int k = 0; k < dist._numComponents; k++) {
                        const div_t tmp = div(k, static_cast<int>(Vecsize));
                        out[i]._weights[k] = dist._weights[tmp.quot][tmp.rem];
                        out[i]._kappas[k] = dist._kappas[tmp.quot][tmp.rem];
                        out[i]._meanDirections[k][0] = dist._meanDirections[tmp.quot].x[tmp.rem];
                        out[i]._meanDirections[k][1] = dist._meanDirections[tmp.quot].y[tmp.rem];
                        out[i]._meanDirections[k][2] = dist._meanDirections[tmp.quot].z[tmp.rem];
                        out[i]._distances[k] = dist._distances[tmp.quot][tmp.rem];
                    }
                    out[i]._pivotPosition[0] = {dist._pivotPosition.x};
                    out[i]._pivotPosition[1] = {dist._pivotPosition.y};
                    out[i]._pivotPosition[2] = {dist._pivotPosition.z};
                    out[i]._numComponents = dist._numComponents;
                }
            }
        }
    */

    void ReleaseFieldData(openpgl::gpu::FieldData *fieldGPU, openpgl::gpu::Device *deviceGPU) const override
    {
        KDTree::NodesType *deviceSurfNodes = (KDTree::NodesType *)fieldGPU->m_surfaceTreeLets;
        delete[] deviceSurfNodes;
        fieldGPU->m_surfaceTreeLets = nullptr;
        fieldGPU->m_numSurfaceTreeLets = 0;

        KDTree::NodesType *deviceVolumeNodes = (KDTree::NodesType *)fieldGPU->m_volumeTreeLets;
        delete[] deviceVolumeNodes;
        fieldGPU->m_volumeTreeLets = nullptr;
        fieldGPU->m_numVolumeTreeLets = 0;

        openpgl::gpu::FlatVMM<32> *outSurf = (openpgl::gpu::FlatVMM<32> *)fieldGPU->m_surfaceDistributions;
        delete[] outSurf;
        fieldGPU->m_surfaceDistributions = nullptr;
        // openpgl::gpu::OutgoingRadianceHistogramData* surfaceOutgoingRadianceHistogram = (openpgl::gpu::OutgoingRadianceHistogramData*)
        // fieldGPU->m_surfaceOutgoingRadianceHistogram; delete[] surfaceOutgoingRadianceHistogram; fieldGPU->m_surfaceOutgoingRadianceHistogram = nullptr;
        fieldGPU->m_numSurfaceDistributions = 0;

        openpgl::gpu::FlatVMM<32> *outVol = (openpgl::gpu::FlatVMM<32> *)fieldGPU->m_volumeDistributions;
        delete[] outVol;
        fieldGPU->m_volumeDistributions = nullptr;

        // openpgl::gpu::OutgoingRadianceHistogramData* volumeOutgoingRadianceHistogram = (openpgl::gpu::OutgoingRadianceHistogramData*)
        // fieldGPU->m_volumeOutgoingRadianceHistogram; delete[] volumeOutgoingRadianceHistogram; fieldGPU->m_volumeOutgoingRadianceHistogram = nullptr;
        fieldGPU->m_numVolumeDistributions = 0;

        openpgl::gpu::VMMPhaseFunctionRepresentationData *pfRep = (openpgl::gpu::VMMPhaseFunctionRepresentationData *)fieldGPU->m_phaseFunctionRepresentations;
        delete[] pfRep;
        fieldGPU->m_phaseFunctionRepresentations = nullptr;
    }

    void FillFieldData(openpgl::gpu::FieldData *fieldGPU, openpgl::gpu::Device *deviceGPU) const override
    {
        int numSurfaceNodes =
        #ifdef USE_TREELETS
            m_surfaceField.m_spatialSubdiv.m_numTreeLets;
        #else
            m_surfaceField.m_spatialSubdiv.m_numNodes;
        #endif
        fieldGPU->m_numSurfaceTreeLets = numSurfaceNodes;
        if (numSurfaceNodes > 0)
        {
            KDTree::NodesType *deviceSurfNodes = new KDTree::NodesType[numSurfaceNodes];
            KDTree::NodesType *src =
            #ifdef USE_TREELETS
                m_surfaceField.m_spatialSubdiv.m_treeLets;
            #else
                m_surfaceField.m_spatialSubdiv.m_nodesPtr;
            #endif
            std::memcpy(deviceSurfNodes, src, numSurfaceNodes * sizeof(KDTree::NodesType));
            // KDTree::NodesType* deviceSurfNodes = deviceGPU->mallocArray<KDTree::NodesType>(numSurfaceNodes);
            // deviceGPU->memcpyArrayToGPU(deviceSurfNodes, m_surfaceField.m_spatialSubdiv.m_treeLets, numSurfaceNodes);

            fieldGPU->m_surfaceTreeLets = (void *)deviceSurfNodes;
        }
        else
        {
            fieldGPU->m_surfaceTreeLets = nullptr;
        }

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
        fieldGPU->m_numPhaseFunctionRepresentations = numPhaseFunctionRepresentations;
        fieldGPU->m_phaseFunctionRepresentations = (void *)phaseFunctionRepresentations;

        int numSurfaceDistriubtion = m_surfaceField.m_regionStorageContainer.size();
        fieldGPU->m_numSurfaceDistributions = numSurfaceDistriubtion;
        if (numSurfaceDistriubtion > 0)
        {
            openpgl::gpu::FlatVMM<32> *outSurf = new openpgl::gpu::FlatVMM<32>[numSurfaceDistriubtion];
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
            openpgl::gpu::OutgoingRadianceHistogramData *outRadianceHistSurf = new openpgl::gpu::OutgoingRadianceHistogramData[numSurfaceDistriubtion];
#endif

            for (int i = 0; i < numSurfaceDistriubtion; i++)
            {
                auto &dist = m_surfaceField.m_regionStorageContainer[i].first.distribution;
                for (int k = 0; k < dist._numComponents; k++)
                {
                    const div_t tmp = div(k, static_cast<int>(Vecsize));
                    outSurf[i]._weights[k] = dist._weights[tmp.quot][tmp.rem];
                    outSurf[i]._kappas[k] = dist._kappas[tmp.quot][tmp.rem];
                    outSurf[i]._meanDirections[k][0] = dist._meanDirections[tmp.quot].x[tmp.rem];
                    outSurf[i]._meanDirections[k][1] = dist._meanDirections[tmp.quot].y[tmp.rem];
                    outSurf[i]._meanDirections[k][2] = dist._meanDirections[tmp.quot].z[tmp.rem];
                    outSurf[i]._distances[k] = dist._distances[tmp.quot][tmp.rem];
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
                    outSurf[i]._fluenceRGBWeights[k][0] = dist._fluenceRGBWeights[tmp.quot].x[tmp.rem];
                    outSurf[i]._fluenceRGBWeights[k][1] = dist._fluenceRGBWeights[tmp.quot].y[tmp.rem];
                    outSurf[i]._fluenceRGBWeights[k][2] = dist._fluenceRGBWeights[tmp.quot].z[tmp.rem];
#endif
                }
                outSurf[i]._pivotPosition[0] = {dist._pivotPosition.x};
                outSurf[i]._pivotPosition[1] = {dist._pivotPosition.y};
                outSurf[i]._pivotPosition[2] = {dist._pivotPosition.z};
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
                outSurf[i]._fluenceRGB[0] = {dist._fluenceRGB.x};
                outSurf[i]._fluenceRGB[1] = {dist._fluenceRGB.y};
                outSurf[i]._fluenceRGB[2] = {dist._fluenceRGB.z};
#endif
                outSurf[i]._numComponents = dist._numComponents;

#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
                auto &outRadianceHist = m_surfaceField.m_regionStorageContainer[i].first.outRadianceHist;
                for (int n = 0; n < OPENPGL_GPU_HISTOGRAM_SIZE; n++)
                {
                    outRadianceHistSurf[i].data[n][0] = outRadianceHist.data[n].x;
                    outRadianceHistSurf[i].data[n][1] = outRadianceHist.data[n].y;
                    outRadianceHistSurf[i].data[n][2] = outRadianceHist.data[n].z;
                }
#endif
                Vector3 rad = dist.outgoingRadiance();
                outSurf[i]._outgoingRGB[0] = rad.x;
                outSurf[i]._outgoingRGB[1] = rad.y;
                outSurf[i]._outgoingRGB[2] = rad.z;
            }

            // openpgl::gpu::FlatVMM<32>* deviceSurf = deviceGPU->mallocArray<openpgl::gpu::FlatVMM<32>>(numSurfaceDistriubtion);
            // deviceGPU->memcpyArrayToGPU(deviceSurf, outSurf, numSurfaceDistriubtion);
            // deviceGPU->wait();
            // fieldGPU->m_surfaceDistributions = (void*) deviceSurf;
            fieldGPU->m_surfaceDistributions = (void *)outSurf;
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
            fieldGPU->m_surfaceOutgoingRadianceHistogram = (void *)outRadianceHistSurf;
#endif
            // delete[] outSurf;
        }
        else
        {
            fieldGPU->m_surfaceDistributions = nullptr;
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
            fieldGPU->m_surfaceOutgoingRadianceHistogram = nullptr;
#endif
        }

        int numVolumeNodes =
        #ifdef USE_TREELETS
            m_volumeField.m_spatialSubdiv.m_numTreeLets;
        #else
            m_volumeField.m_spatialSubdiv.m_numNodes;
        #endif
        fieldGPU->m_numVolumeTreeLets = numVolumeNodes;
        if (numVolumeNodes > 0)
        {
            KDTree::NodesType *deviceVolumeNodes = new KDTree::NodesType[numVolumeNodes];
            KDTree::NodesType *src =
            #ifdef USE_TREELETS
                m_volumeField.m_spatialSubdiv.m_treeLets;
            #else
                m_volumeField.m_spatialSubdiv.m_nodesPtr;
            #endif
            std::memcpy(deviceVolumeNodes, src, numVolumeNodes * sizeof(KDTree::NodesType));
            // KDTree::NodesType* deviceVolumeNodes = deviceGPU->mallocArray<KDTree::NodesType>(numVolumeNodes);
            // deviceGPU->memcpyArrayToGPU(deviceVolumeNodes, m_volumeField.m_spatialSubdiv.m_treeLets, numVolumeNodes);
            fieldGPU->m_volumeTreeLets = (void *)deviceVolumeNodes;
        }
        else
        {
            fieldGPU->m_volumeTreeLets = nullptr;
        }

        int numVolumeDistriubtion = m_volumeField.m_regionStorageContainer.size();
        fieldGPU->m_numVolumeDistributions = numVolumeDistriubtion;
        if (numVolumeDistriubtion > 0)
        {
            openpgl::gpu::FlatVMM<32> *outVol = new openpgl::gpu::FlatVMM<32>[numVolumeDistriubtion];
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
            openpgl::gpu::OutgoingRadianceHistogramData *outRadianceHistVol = new openpgl::gpu::OutgoingRadianceHistogramData[numVolumeDistriubtion];
#endif
            for (int i = 0; i < numVolumeDistriubtion; i++)
            {
                auto &dist = m_volumeField.m_regionStorageContainer[i].first.distribution;
                for (int k = 0; k < dist._numComponents; k++)
                {
                    const div_t tmp = div(k, static_cast<int>(Vecsize));
                    outVol[i]._weights[k] = dist._weights[tmp.quot][tmp.rem];
                    outVol[i]._kappas[k] = dist._kappas[tmp.quot][tmp.rem];
                    outVol[i]._meanDirections[k][0] = dist._meanDirections[tmp.quot].x[tmp.rem];
                    outVol[i]._meanDirections[k][1] = dist._meanDirections[tmp.quot].y[tmp.rem];
                    outVol[i]._meanDirections[k][2] = dist._meanDirections[tmp.quot].z[tmp.rem];
                    outVol[i]._distances[k] = dist._distances[tmp.quot][tmp.rem];
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
                    outVol[i]._fluenceRGBWeights[k][0] = dist._fluenceRGBWeights[tmp.quot].x[tmp.rem];
                    outVol[i]._fluenceRGBWeights[k][1] = dist._fluenceRGBWeights[tmp.quot].y[tmp.rem];
                    outVol[i]._fluenceRGBWeights[k][2] = dist._fluenceRGBWeights[tmp.quot].z[tmp.rem];
#endif
                }
                outVol[i]._pivotPosition[0] = {dist._pivotPosition.x};
                outVol[i]._pivotPosition[1] = {dist._pivotPosition.y};
                outVol[i]._pivotPosition[2] = {dist._pivotPosition.z};
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
                outVol[i]._fluenceRGB[0] = {dist._fluenceRGB.x};
                outVol[i]._fluenceRGB[1] = {dist._fluenceRGB.y};
                outVol[i]._fluenceRGB[2] = {dist._fluenceRGB.z};
#endif
                outVol[i]._numComponents = dist._numComponents;
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
                auto &outRadianceHist = m_volumeField.m_regionStorageContainer[i].first.outRadianceHist;
                for (int n = 0; n < OPENPGL_GPU_HISTOGRAM_SIZE; n++)
                {
                    outRadianceHistVol[i].data[n][0] = outRadianceHist.data[n].x;
                    outRadianceHistVol[i].data[n][1] = outRadianceHist.data[n].y;
                    outRadianceHistVol[i].data[n][2] = outRadianceHist.data[n].z;
                }
#endif
                Vector3 rad = dist.outgoingRadiance();
                outVol[i]._outgoingRGB[0] = rad.x;
                outVol[i]._outgoingRGB[1] = rad.y;
                outVol[i]._outgoingRGB[2] = rad.z;
            }

            // openpgl::gpu::FlatVMM<32>* deviceVol = deviceGPU->mallocArray<openpgl::gpu::FlatVMM<32>>(numVolumeDistriubtion);
            // deviceGPU->memcpyArrayToGPU(deviceVol, outVol, numVolumeDistriubtion);
            // deviceGPU->wait();
            // delete[] outVol;
            fieldGPU->m_volumeDistributions = (void *)outVol;
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
            fieldGPU->m_volumeOutgoingRadianceHistogram = (void *)outRadianceHistVol;
#endif
        }
        else
        {
            fieldGPU->m_volumeDistributions = nullptr;
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
            fieldGPU->m_volumeOutgoingRadianceHistogram = nullptr;
#endif
        }
    }

    openpgl::gpu::FieldData GetFieldGPU(openpgl::gpu::Device *deviceGPU)
    {
        openpgl::gpu::FieldData field;
        FillFieldData(&field, deviceGPU);
        return field;
    }

    PGLRange getSurfaceSampleRange(size_t id) const override
    {
        return m_surfaceField.getSampleRange(id);
    }

    PGLRange getVolumeSampleRange(size_t id) const override
    {
        return m_volumeField.getSampleRange(id);
    }

    //void runUpdateDump(const std::string updateDumpFilename, const bool surface = true) const override
    //{
    //    if (surface)
    //        m_surfaceField.runUpdateDump(updateDumpFilename, true);
    //    else
    //        m_volumeField.runUpdateDump(updateDumpFilename, false);
    //}

    void sDump(SDump *sDump) const override {
        sDump->sur = new SDumpTree;
        m_surfaceField.sDump(sDump->sur);
        sDump->vol = new SDumpTree;
        m_volumeField.sDump(sDump->vol);
    }

   private:
    size_t m_iteration{0};
    size_t m_totalSPP{0};

    FieldType m_surfaceField;
    FieldType m_volumeField;
};

}
}  // namespace openpgl
