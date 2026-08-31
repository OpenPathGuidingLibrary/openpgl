#include "Device.h"

struct PathSegment{
    bool isDelta;
    bool volumeScatter;
    Point3 position;
    Normal3 normal;
    Vector3 scatterWeight;
    Vector3 directionIn;
    float pdfDirectionIn;
    Vector3 directionOut;
    Vector3 transmittanceWeight;
    Vector3 directContribution;
    Vector3 scatteredContribution;
    float miWeight;
    float rrProbability;
    float eta;
    float roughness;
};

enum {
    EPathSegmentStorageDepth = 11,
};

struct PathSegmentStorage{
    PathSegment segments[EPathSegmentStorageDepth];
    uint32_t curDepth;
};


#include "PathSegmentStorage_soa.h"

struct PathSegmentStorageBuffer: public SOA<PathSegmentStorage> {

public:

    uint32_t max_depth;
    PathSegmentStorageBuffer() = default;
    PathSegmentStorageBuffer(int n, openpgl::gpu::Device *device) : SOA<PathSegmentStorage>(n, device), max_depth(EPathSegmentStorageDepth){
        //std::cout << "PathSegmentStorage(int n, openpgl::gpu::Device *device)" << std::endl;
    }

    PathSegmentStorageBuffer(std::string fileName, openpgl::gpu::Device *device)
    {
        std::filebuf fb;
        fb.open(fileName, std::ios::in | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);
        this->deserialize(is, device);
        fb.close();
    }



    void Store(std::string fileName) const {
        std::filebuf fb;
        fb.open(fileName, std::ios::out | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file!");
        std::ostream os(&fb);
        this->serialize(os);
        fb.close();
    }

    OPENPGL_GPU_CALLABLE
    void PropagateSamples(const int pixelIndex, const SampleDataStorageBuffer* sampleDataStorageBuffer) const {
        int depth = curDepth[pixelIndex];
        Vector3 contribution(0.f, 0.f , 0.f);

        Vector3 scatterWeights[EPathSegmentStorageDepth];
        Point3 positions[EPathSegmentStorageDepth];
        Vector3 directions[EPathSegmentStorageDepth];
        float pdfs[EPathSegmentStorageDepth];
        Vector3 contributions[EPathSegmentStorageDepth];
        float roughensses[EPathSegmentStorageDepth];
        bool isDeltas[EPathSegmentStorageDepth];

        for (int n = 0; n < depth; n++) {
            scatterWeights[n] = segments[n].scatterWeight[pixelIndex];
            positions[n] = segments[n].position[pixelIndex];
            directions[n] = segments[n].directionIn[pixelIndex];
            pdfs[n] = segments[n].pdfDirectionIn[pixelIndex];
            contributions[n] = segments[n].scatteredContribution[pixelIndex];
            contributions[n] += segments[n].directContribution[pixelIndex];
            roughensses[n] = segments[n].roughness[pixelIndex];
            isDeltas[n] = segments[n].isDelta[pixelIndex];
        }

        for (int n = depth-2; n >= 0; n--) {
            Vector3 dist;
            dist[0] = positions[n+1][0] - positions[n][0];
            dist[1] = positions[n+1][1] - positions[n][1];
            dist[2] = positions[n+1][2] - positions[n][2];
            float distance = length(dist);
            distance = 1e6f;
            contribution += contributions[n+1];
            if ((contribution[0] > 0.f || contribution[1] > 0.f || contribution[2] > 0.f) && (!isDeltas[n]) && roughensses[n] > 0.1f) {
                sampleDataStorageBuffer->AddSampleData(pixelIndex, positions[n], directions[n], pdfs[n], distance, contribution, {0, 0, 0}, false);
            }
            //else 
            //{
            ////} else if (contribution[0] == -1.f){
            //    sampleDataStorageBuffer.AddZeroValueSampleData(pixelIndex, pos, direction, false);
            //}
            contribution[0] *= scatterWeights[n][0];
            contribution[1] *= scatterWeights[n][1];
            contribution[2] *= scatterWeights[n][2];
        }
    }

    OPENPGL_GPU_CALLABLE
    void Reset(const int pixelIndex) const {
        curDepth[pixelIndex] = 0;
    }

    void PrepareSampleData(SampleDataStorageBuffer& sampleDataStorageBuffer) {
#if defined(OPENPGL_GPU_CUDA)
        OPENPGL_CUDA_CHECK(cudaDeviceSynchronize());
#endif
        uint32_t maxQueueSize = this->nAlloc; 
        ParallelFor(device, "PrepareSampleData", maxQueueSize, OPENPGL_CPU_GPU_LAMBDA(int pixelIndex) {
                PropagateSamples(pixelIndex, &sampleDataStorageBuffer);
            }
        );
    }

    void Reset() {
#if defined(OPENPGL_GPU_CUDA)
        OPENPGL_CUDA_CHECK(cudaDeviceSynchronize());
#endif
        uint32_t maxQueueSize = this->nAlloc; 
        ParallelFor(device, "Reset", maxQueueSize, OPENPGL_CPU_GPU_LAMBDA(int pixelIndex) {
                Reset(pixelIndex);
            }
        );
    }
};