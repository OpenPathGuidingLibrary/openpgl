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