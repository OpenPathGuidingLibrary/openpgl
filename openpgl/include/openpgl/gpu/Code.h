#if defined(OPENPGL_GPU_SYCL)
#ifdef fmaxf
    #undef fmaxf
#endif
#ifdef fminf
#undef fminf
#endif
#endif
struct Region
{
   public:
};

struct KDNode
{
    enum
    {
        ESPlitDimX = 0,
        ESPlitDimY = 1,
        ESPlitDimZ = 2,
        ELeafNode = 3,
    };

    float splitPosition{0.0f};
    uint32_t splitDimAndNodeIdx{0};

    /////////////////////////////
    // Child node functions
    /////////////////////////////
    bool isChild() const
    {
        return (splitDimAndNodeIdx >> 30) < ELeafNode;
    }

    void setLeftChildIdx(const uint32_t &idx)
    {
        // OPENPGL_ASSERT(idx < (1U << 31));
        // OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
        // OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
        // OPENPGL_ASSERT(!isLeaf());

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
        // OPENPGL_ASSERT(idx == getLeftChildIdx());
    }

    OPENPGL_GPU_CALLABLE uint32_t getLeftChildIdx() const
    {
        // OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
        // OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
        // OPENPGL_ASSERT(!isLeaf());

        return (splitDimAndNodeIdx << 2) >> 2;
    }

    /////////////////////////////
    // Inner node functions
    /////////////////////////////

    void setToInnerNode(const uint8_t &_splitDim, const float &_splitPos, const uint32_t &_leftChildIdx)
    {
        splitPosition = _splitPos;
        splitDimAndNodeIdx = 0;
        splitDimAndNodeIdx = (uint32_t(_splitDim) << 30);
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | _leftChildIdx;

        // OPENPGL_ASSERT(_splitDim == getSplitDim());
        // OPENPGL_ASSERT(_leftChildIdx == getLeftChildIdx());
    }

    /////////////////////////////
    // Leaf node functions
    /////////////////////////////

    void setLeaf()
    {
        splitDimAndNodeIdx = (3U << 30);
        // OPENPGL_ASSERT(isLeaf());
    }

    OPENPGL_GPU_CALLABLE bool isLeaf() const
    {
        return (splitDimAndNodeIdx >> 30) == 3;
    }

    void setChildNodeIdx(const uint32_t &idx)
    {
        // OPENPGL_ASSERT(idx < (1U << 31));
        // OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
        // OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
        // OPENPGL_ASSERT(!isLeaf());

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
        // OPENPGL_ASSERT(idx == getLeftChildIdx());
    }

    void setDataNodeIdx(const uint32_t &idx)
    {
        // OPENPGL_ASSERT(idx < (1U << 31)); // checks if the idx is in the right range
        setLeaf();
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
        // OPENPGL_ASSERT(isLeaf());
        // OPENPGL_ASSERT(getDataIdx() == idx);
    }

    OPENPGL_GPU_CALLABLE uint32_t getDataIdx() const
    {
        // OPENPGL_ASSERT(isLeaf());
        return (splitDimAndNodeIdx << 2) >> 2;
    }

    /////////////////////////////
    // Split dimension functions
    /////////////////////////////

    OPENPGL_GPU_CALLABLE uint8_t getSplitDim() const
    {
        return (splitDimAndNodeIdx >> 30);
    }

    void setSplitDim(const uint8_t &splitAxis)
    {
        // OPENPGL_ASSERT(splitAxis < ELeafNode);
        splitDimAndNodeIdx = (uint32_t(splitAxis) << 30);
        // OPENPGL_ASSERT(splitAxis == getSplitDim());
    }

    OPENPGL_GPU_CALLABLE float getSplitPivot() const
    {
        return splitPosition;
    }

    void setSplitPivot(const float &pos)
    {
        splitPosition = pos;
    }

    void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&splitPosition), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&splitDimAndNodeIdx), sizeof(uint32_t));
    }

    void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&splitPosition), sizeof(float));
        stream.read(reinterpret_cast<char *>(&splitDimAndNodeIdx), sizeof(uint32_t));
    }

    bool operator==(const KDNode &b) const
    {
        bool equal = true;
        if (splitPosition != b.splitPosition || isLeaf() != b.isLeaf())
        {
            equal = false;
        }
        if (!isLeaf())
        {
            if (getSplitDim() != b.getSplitDim() || getLeftChildIdx() != b.getLeftChildIdx())
            {
                equal = false;
            }
        }
        else
        {
            if (getDataIdx() != b.getDataIdx())
            {
                equal = false;
            }
        }
        return equal;
    }
};

struct KDTreeLet
{
    KDNode nodes[8];
};

struct FieldGPU : public FieldData
{
    using Distribution = ParallaxAwareVonMisesFisherMixture<32>;

    OPENPGL_GPU_CALLABLE FieldGPU() = default;

    OPENPGL_GPU_CALLABLE FieldGPU(const FieldGPU& field){
        this->m_ready = field.m_ready;
        this->m_numSurfaceTreeLets = field.m_numSurfaceTreeLets;
        this->m_numVolumeTreeLets = field.m_numVolumeTreeLets;

        this->m_surfaceTreeLets = field.m_surfaceTreeLets;
        this->m_volumeTreeLets = field.m_volumeTreeLets;

        this->m_numSurfaceDistributions = field.m_numSurfaceDistributions;
        this->m_numVolumeDistributions = field.m_numVolumeDistributions;

        this->m_surfaceDistributions = field.m_surfaceDistributions;
        this->m_volumeDistributions = field.m_volumeDistributions;

        this->m_numPhaseFunctionRepresentations = field.m_numPhaseFunctionRepresentations;
        this->m_phaseFunctionRepresentations = field.m_phaseFunctionRepresentations;

#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
        this->m_surfaceOutgoingRadianceHistogram = field.m_surfaceOutgoingRadianceHistogram;
        this->m_volumeOutgoingRadianceHistogram = field.m_volumeOutgoingRadianceHistogram;
#endif
    };

    FieldGPU(openpgl::gpu::Device *device)
    {
        (void)device;
        this->m_ready = false;
        this->m_numSurfaceTreeLets = 0;
        this->m_numVolumeTreeLets = 0;

        this->m_surfaceTreeLets = nullptr;
        this->m_volumeTreeLets = nullptr;

        this->m_numSurfaceDistributions = 0;
        this->m_numVolumeDistributions = 0;

        this->m_surfaceDistributions = nullptr;
        this->m_volumeDistributions = nullptr;

        this->m_numPhaseFunctionRepresentations = 0;
        this->m_phaseFunctionRepresentations = nullptr;

#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
        this->m_surfaceOutgoingRadianceHistogram = nullptr;
        this->m_volumeOutgoingRadianceHistogram = nullptr;
#endif
    }

    FieldGPU(openpgl::gpu::Device *device, const openpgl::cpp::Field *field)
    {
        FieldData fieldData;
        field->FillFieldGPU(&fieldData, device);
        m_numSurfaceTreeLets = fieldData.m_numSurfaceTreeLets;
        if (m_numSurfaceTreeLets > 0)
        {
            m_surfaceTreeLets = (KDTreeLet *)device->mallocArray<KDTreeLet>(m_numSurfaceTreeLets);
            device->memcpyArrayToGPU((KDTreeLet *)m_surfaceTreeLets, (KDTreeLet *)fieldData.m_surfaceTreeLets, m_numSurfaceTreeLets);
        }
        else
        {
            m_surfaceTreeLets = nullptr;
        }

        m_numPhaseFunctionRepresentations = fieldData.m_numPhaseFunctionRepresentations;
        if (m_numPhaseFunctionRepresentations > 0)
        {
            m_phaseFunctionRepresentations = device->mallocArray<VMMPhaseFunctionRepresentationData>(m_numPhaseFunctionRepresentations);
            device->memcpyArrayToGPU((VMMPhaseFunctionRepresentationData *)m_phaseFunctionRepresentations,
                                     (const VMMPhaseFunctionRepresentationData *)fieldData.m_phaseFunctionRepresentations, m_numPhaseFunctionRepresentations);
        }
        else
        {
            m_phaseFunctionRepresentations = nullptr;
        }

        m_numSurfaceDistributions = fieldData.m_numSurfaceDistributions;
        if (m_numSurfaceDistributions > 0)
        {
            m_surfaceDistributions = device->mallocArray<Distribution>(m_numSurfaceDistributions);
            device->memcpyArrayToGPU((Distribution *)m_surfaceDistributions, (const Distribution *)fieldData.m_surfaceDistributions, m_numSurfaceDistributions);
#ifdef OPENPGL_EF_RADIANCE_CACHES
            m_surfaceOutgoingRadianceHistogram = device->mallocArray<OutgoingRadianceHistogramData>(m_numSurfaceDistributions);
            device->memcpyArrayToGPU((OutgoingRadianceHistogramData *)m_surfaceOutgoingRadianceHistogram,
                                     (const OutgoingRadianceHistogramData *)fieldData.m_surfaceOutgoingRadianceHistogram, m_numSurfaceDistributions);
#endif
        }
        else
        {
            m_surfaceDistributions = nullptr;
        }

        m_numVolumeTreeLets = fieldData.m_numVolumeTreeLets;
        if (m_numVolumeTreeLets > 0)
        {
            m_volumeTreeLets = (KDTreeLet *)device->mallocArray<KDTreeLet>(m_numVolumeTreeLets);
            device->memcpyArrayToGPU((KDTreeLet *)m_volumeTreeLets, (KDTreeLet *)fieldData.m_volumeTreeLets, m_numVolumeTreeLets);
        }
        else
        {
            m_volumeTreeLets = nullptr;
        }

        m_numVolumeDistributions = fieldData.m_numVolumeDistributions;
        if (m_numVolumeDistributions > 0)
        {
            m_volumeDistributions = device->mallocArray<Distribution>(m_numVolumeDistributions);
            device->memcpyArrayToGPU((Distribution *)m_volumeDistributions, (const Distribution *)fieldData.m_volumeDistributions, m_numVolumeDistributions);
#ifdef OPENPGL_EF_RADIANCE_CACHES
            m_volumeOutgoingRadianceHistogram = device->mallocArray<OutgoingRadianceHistogramData>(m_numVolumeDistributions);
            device->memcpyArrayToGPU((OutgoingRadianceHistogramData *)m_volumeOutgoingRadianceHistogram,
                                     (const OutgoingRadianceHistogramData *)fieldData.m_volumeOutgoingRadianceHistogram, m_numVolumeDistributions);
#endif
        }
        else
        {
            m_volumeDistributions = nullptr;
        }
        device->wait();
        field->ReleaseFieldGPU(&fieldData, device);

        if ((m_numSurfaceTreeLets > 0 && m_numSurfaceDistributions > 0) || (m_numVolumeTreeLets > 0 && m_numVolumeDistributions > 0))
            this->m_ready = true;
        else 
            this->m_ready = false;
        /*
        int numSurfaceNodes = field->GetNumSurfaceKDNodes();
        if(numSurfaceNodes > 0) {
            m_surfaceTreeLets = (KDTreeLet*) device->mallocArray<KDTreeLet>(numSurfaceNodes);
            device->memcpyArrayToGPU((KDTreeLet*) m_surfaceTreeLets, (KDTreeLet*)field->GetSurfaceKdNodes(), numSurfaceNodes);
        }
        int numSurfaceDistributions = field->GetNumSurfaceDistributions();
        Distribution *surfaceDistributions = nullptr;
        if(numSurfaceDistributions > 0) {
            surfaceDistributions = new Distribution[numSurfaceDistributions];
            m_surfaceDistributions = device->mallocArray<Distribution>(numSurfaceDistributions);
            field->CopySurfaceDistributions(surfaceDistributions);
            device->memcpyArrayToGPU((Distribution*) m_surfaceDistributions, (const Distribution*) surfaceDistributions, numSurfaceDistributions);
        }

        int numVolumeNodes = field->GetNumVolumeKDNodes();
        if(numVolumeNodes > 0) {
            m_volumeTreeLets = (KDTreeLet*) device->mallocArray<KDTreeLet>(numVolumeNodes);
            device->memcpyArrayToGPU((KDTreeLet*) m_volumeTreeLets, (KDTreeLet*)field->GetVolumeKdNodes(), numVolumeNodes);
        }
        int numVolumeDistributions = field->GetNumVolumeDistributions();
        Distribution *volumeDistributions = nullptr;
        if(numVolumeDistributions > 0) {
            volumeDistributions = new Distribution[numVolumeDistributions];
            m_volumeDistributions = device->mallocArray<Distribution>(numVolumeDistributions);
            field->CopyVolumeDistributions(volumeDistributions);
            device->memcpyArrayToGPU((Distribution*) m_volumeDistributions, (const Distribution*) volumeDistributions, numVolumeDistributions);
        }

        // wait until all CPU -> GPU copies are done
        device->wait();
        delete[] surfaceDistributions;
        delete[] volumeDistributions;
        */
    }

    void Release(openpgl::gpu::Device *device) {
        if (m_numSurfaceTreeLets > 0 && m_surfaceTreeLets != nullptr)
        {
            device->freeArray<KDTreeLet>((KDTreeLet*)m_surfaceTreeLets);
        }
        m_numSurfaceTreeLets = 0;
        m_surfaceTreeLets = nullptr;

        if (m_numPhaseFunctionRepresentations > 0 && m_phaseFunctionRepresentations != nullptr)
        {
            device->freeArray<VMMPhaseFunctionRepresentationData>((VMMPhaseFunctionRepresentationData*)m_phaseFunctionRepresentations);
        }
        m_numPhaseFunctionRepresentations = 0;
        m_phaseFunctionRepresentations = nullptr;

        if (m_numSurfaceDistributions > 0 && m_surfaceDistributions != nullptr)
        {
            device->freeArray<Distribution>((Distribution*)m_surfaceDistributions);
#ifdef OPENPGL_EF_RADIANCE_CACHES
            device->freeArray<OutgoingRadianceHistogramData>((OutgoingRadianceHistogramData*)m_surfaceOutgoingRadianceHistogram);
#endif
        }

        m_numSurfaceDistributions = 0;
        m_surfaceDistributions = nullptr;
#ifdef OPENPGL_EF_RADIANCE_CACHES
        m_surfaceOutgoingRadianceHistogram = nullptr;
#endif

        if (m_numVolumeTreeLets > 0 && m_volumeTreeLets != nullptr)
        {
            device->freeArray<KDTreeLet>((KDTreeLet*)m_volumeTreeLets);
        }

        m_numVolumeTreeLets = 0;
        m_volumeTreeLets = nullptr;

        if (m_numVolumeDistributions > 0 && m_volumeDistributions != nullptr)
        {
            device->freeArray<Distribution>((Distribution*)m_volumeDistributions);
#ifdef OPENPGL_EF_RADIANCE_CACHES
            device->freeArray<OutgoingRadianceHistogramData>((OutgoingRadianceHistogramData*)m_volumeOutgoingRadianceHistogram);
#endif
        }
        m_numVolumeDistributions = 0;
#ifdef OPENPGL_EF_RADIANCE_CACHES
        m_volumeOutgoingRadianceHistogram = nullptr;
#endif
        m_volumeDistributions = nullptr;
    }

    OPENPGL_GPU_CALLABLE uint32_t getDataIdxAtPos(const float *pos, const KDTreeLet *treeLets) const
    {
#ifdef USE_TREELETS
        uint32_t treeIdx = 0;
        uint32_t nodeIdx = 0;
        uint32_t depth = 0;
        KDTreeLet treeLet = treeLets[treeIdx];

        while (!treeLet.nodes[nodeIdx].isLeaf())
        {
            uint8_t splitDim = treeLet.nodes[nodeIdx].getSplitDim();
            uint32_t childIdx = treeLet.nodes[nodeIdx].getLeftChildIdx();
            float pivot = treeLet.nodes[nodeIdx].getSplitPivot();

            if (depth % 3 == 2)
            {
                nodeIdx = 0;
                treeIdx = childIdx;
                treeIdx += pos[splitDim] >= pivot ? 1 : 0;
                treeLet = treeLets[treeIdx];
            }
            else
            {
                nodeIdx = childIdx - (treeIdx * 8);
                nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
            }
            depth++;
        }
        return treeLet.nodes[nodeIdx].getDataIdx();
#else
        uint32_t nodeIdx = 0;
        while (!m_nodesPtr[nodeIdx].isLeaf())
        {
            uint8_t splitDim = m_nodesPtr[nodeIdx].getSplitDim();
            float pivot = m_nodesPtr[nodeIdx].getSplitPivot();

            nodeIdx = m_nodesPtr[nodeIdx].getLeftChildIdx();
            nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
        }
        return m_nodesPtr[nodeIdx].getDataIdx();
#endif
    }

    OPENPGL_GPU_CALLABLE uint32_t getSurfaceDistributionIdxAtPos(const float *pos) const
    {
        return getDataIdxAtPos(pos, (const KDTreeLet *)m_surfaceTreeLets);
    }

    OPENPGL_GPU_CALLABLE uint32_t getVolumeDistributionIdxAtPos(const float *pos) const
    {
        return getDataIdxAtPos(pos, (const KDTreeLet *)m_volumeTreeLets);
    }

    OPENPGL_GPU_CALLABLE VMMPhaseFunctionRepresentationData GetHenyeyGreensteinPhaseFunctionRepresentation(const float g) const
    {
        const float meanCosine = g;
        const float absMeanCosine = std::fabs(meanCosine);
        const float stepSize = (0.99f - 0.f) / float(m_numPhaseFunctionRepresentations);
        int idx = std::floor((absMeanCosine - 0.f) / stepSize);
        idx = std::fminf(idx, m_numPhaseFunctionRepresentations - 1);

        return ((VMMPhaseFunctionRepresentationData *)m_phaseFunctionRepresentations)[idx];
    }

    OPENPGL_GPU_CALLABLE void Reset() {}

    OPENPGL_GPU_CALLABLE bool IsReady() const {return m_ready;}

    // #ifdef USE_TREELETS
    /* Need to disable these direct initializations to pass through the std::is_trivially_default_constructible_v test. If not the following error is throughen:
     *  error: static assertion failed due to requirement 'std::is_trivially_default_constructible_v<const openpgl_gpu::FieldGPU>': Type T must be trivially default constructable
     * (until C++20 consteval is supported and enabled.)
     */
    /*

            void *m_surfaceTreeLets;//{nullptr};
            void *m_volumeTreeLets;//{nullptr};
    #else
            KDNode *m_surfaceNodesPtr;//{nullptr};
    #endif
            void *m_surfaceDistributions;// {nullptr};
            void *m_volumeDistributions;// {nullptr};
    */
};
/*
struct SurfaceSamplingDistributionData
{
    OPENPGL_GPU_CALLABLE SurfaceSamplingDistributionData() = default;
    const FieldGPU *m_field{nullptr};
    pgl_point3f m_pos{0.f, 0.f, 0.f};
    int m_idx{-1};
};
*/
struct SurfaceSamplingDistribution : public SurfaceSamplingDistributionData
{
    OPENPGL_GPU_CALLABLE SurfaceSamplingDistribution() = default;

    OPENPGL_GPU_CALLABLE bool Init(const FieldGPU *field, const pgl_point3f &pos, float &sample1D)
    {
        (void)sample1D;
        if(!field->IsReady())
            return false;
        m_pos = pos;
        m_field = field;
        float _pos[3] = {pos.x, pos.y, pos.z};
        m_idx = ((const FieldGPU*) m_field)->getSurfaceDistributionIdxAtPos(_pos);

        return m_idx >= 0;
    }

    OPENPGL_GPU_CALLABLE void Clear()
    {
        m_field = nullptr;
        m_pos = {0.f, 0.f, 0.f};
        m_idx = -1;
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f Sample(const pgl_point2f &sample2D) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *surfaceDistributions = static_cast<const FieldGPU::Distribution *>(field->m_surfaceDistributions);
        return surfaceDistributions[m_idx].samplePos(m_pos, sample2D);
    }

    OPENPGL_GPU_CALLABLE float PDF(const pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *surfaceDistributions = static_cast<const FieldGPU::Distribution *>(field->m_surfaceDistributions);
        return surfaceDistributions[m_idx].pdfPos(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE float SamplePDF(const pgl_point2f &sample2D, pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *surfaceDistributions = static_cast<const FieldGPU::Distribution *>(field->m_surfaceDistributions);
        direction = surfaceDistributions[m_idx].samplePos(m_pos, sample2D);
        return surfaceDistributions[m_idx].pdfPos(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE float IncomingRadiancePDF(const pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *surfaceDistributions = static_cast<const FieldGPU::Distribution *>(field->m_surfaceDistributions);
        return surfaceDistributions[m_idx].pdfPos(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE bool SupportsApplyCosineProduct() const
    {
        return false;
    }

    OPENPGL_GPU_CALLABLE uint32_t GetId() const
    {
        return m_idx;
    }

#ifdef OPENPGL_EF_RADIANCE_CACHES
    OPENPGL_GPU_CALLABLE pgl_vec3f IncomingRadiance(pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *surfaceDistributions = static_cast<const FieldGPU::Distribution *>(field->m_surfaceDistributions);
        return surfaceDistributions[m_idx].incomingRadiance(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f Irradiance(pgl_vec3f &normal) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *surfaceDistributions = static_cast<const FieldGPU::Distribution *>(field->m_surfaceDistributions);
        return surfaceDistributions[m_idx].irradiance(m_pos, normal);
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f OutgoingRadiance(pgl_vec3f &direction) const
    {
        const OutgoingRadianceHistogramData *outgoingRadianceHistograms = static_cast<const OutgoingRadianceHistogramData *>(m_field->m_surfaceOutgoingRadianceHistogram);
        const Vector3 dir = {direction.x, direction.y, direction.z};
        const pgl_vec2f p = directionToCanonical(dir);
        const int res = OPENPGL_GPU_HISTOGRAM_RESOLUTION;
        const int histIdx = std::min(int(p.x * res), res - 1) + std::min(int(p.y * res), res - 1) * res;
        return {outgoingRadianceHistograms[m_idx].data[histIdx][0], outgoingRadianceHistograms[m_idx].data[histIdx][1], outgoingRadianceHistograms[m_idx].data[histIdx][2]};
    }
#endif
};

/*
struct VolumeSamplingDistributionData
{
    OPENPGL_GPU_CALLABLE VolumeSamplingDistributionData() = default;
    const FieldGPU *m_field{nullptr};
    pgl_point3f m_pos{0.f, 0.f, 0.f};
    int m_idx{-1};
    VMMPhaseFunctionRepresentationData m_phaseRep;
};
*/
struct VolumeSamplingDistribution : public VolumeSamplingDistributionData
{
    OPENPGL_GPU_CALLABLE VolumeSamplingDistribution() = default;

    OPENPGL_GPU_CALLABLE bool Init(const FieldGPU *field, const pgl_point3f &pos, float &sample1D)
    {
        (void)sample1D;
        if(!field->IsReady())
            return false;
        m_pos = pos;
        m_field = field;
        float _pos[3] = {pos.x, pos.y, pos.z};
        m_idx = field->getVolumeDistributionIdxAtPos(_pos);

        return m_idx >= 0;
    }

    OPENPGL_GPU_CALLABLE void SetPhaseFunction(const float g)
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        m_phaseRep = field->GetHenyeyGreensteinPhaseFunctionRepresentation(g);
    }

    OPENPGL_GPU_CALLABLE void Clear()
    {
        m_field = nullptr;
        m_pos = {0.f, 0.f, 0.f};
        m_idx = -1;
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f Sample(const pgl_point2f &sample2D) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *volumeDistributions = static_cast<const FieldGPU::Distribution *>(field->m_volumeDistributions);
        return volumeDistributions[m_idx].samplePos(m_pos, sample2D);
    }

    OPENPGL_GPU_CALLABLE float PDF(const pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *volumeDistributions = static_cast<const FieldGPU::Distribution *>(field->m_volumeDistributions);
        return volumeDistributions[m_idx].pdfPos(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE float SamplePDF(const pgl_point2f &sample2D, pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *volumeDistributions = static_cast<const FieldGPU::Distribution *>(field->m_volumeDistributions);
        direction = volumeDistributions[m_idx].samplePos(m_pos, sample2D);
        return volumeDistributions[m_idx].pdfPos(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE float IncomingRadiancePDF(const pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *volumeDistributions = static_cast<const FieldGPU::Distribution *>(field->m_volumeDistributions);
        return volumeDistributions[m_idx].pdfPos(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE uint32_t GetId() const
    {
        return m_idx;
    }

#ifdef OPENPGL_EF_RADIANCE_CACHES
    OPENPGL_GPU_CALLABLE pgl_vec3f IncomingRadiance(pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *volumeDistributions = static_cast<const FieldGPU::Distribution *>(field->m_volumeDistributions);
        return volumeDistributions[m_idx].incomingRadiance(m_pos, direction);
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f OutgoingRadiance(pgl_vec3f &direction) const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const OutgoingRadianceHistogramData *outgoingRadianceHistograms = static_cast<const OutgoingRadianceHistogramData *>(field->m_volumeOutgoingRadianceHistogram);
        const Vector3 dir = {direction.x, direction.y, direction.z};
        const pgl_vec2f p = directionToCanonical(dir);
        const int res = OPENPGL_GPU_HISTOGRAM_RESOLUTION;
        const int histIdx = std::min(int(p.x * res), res - 1) + std::min(int(p.y * res), res - 1) * res;
        return {outgoingRadianceHistograms[m_idx].data[histIdx][0], outgoingRadianceHistograms[m_idx].data[histIdx][1], outgoingRadianceHistograms[m_idx].data[histIdx][2]};
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f Fluence() const
    {
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *volumeDistributions = static_cast<const FieldGPU::Distribution *>(field->m_volumeDistributions);
        return volumeDistributions[m_idx].fluence();
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f InscatteredRadiance(pgl_vec3f &direction, const float g) const
    {
        // TODO: lookup VMF mean cosine for HG mean cosine
        const FieldGPU* field = static_cast<const FieldGPU *>(m_field);
        const FieldGPU::Distribution *volumeDistributions = static_cast<const FieldGPU::Distribution *>(field->m_volumeDistributions);
        return volumeDistributions[m_idx].inscatteredRadiance(m_pos, direction, m_phaseRep);
    }
#endif
};