#include "Device.h"
#include "../cpp/SampleData.h"


#define USE_RAW_SAMPLE_DATA

class Timer
{
   private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

   public:
    Timer()
    {
        reset();
    }

    void reset()
    {
        start = clock::now();
    }

    double elapsed()
    {
        time_point end = clock::now();
        std::chrono::duration<double, std::micro> diff = end - start;
        return diff.count();
    }

   private:
    time_point start;
};

#ifndef USE_RAW_SAMPLE_DATA

struct SampleData{
    uint32_t flags;
    Point3 position;
    Vector3 direction;
    float weight;
    float pdf;
    float distance;
};

struct ZeroValueSampleData{
    bool volume;
    Point3 position;
    Vector3 direction;
};

#else

using SampleData = openpgl::cpp::SampleData;
using ZeroValueSampleData = openpgl::cpp::ZeroValueSampleData;

#endif

enum {
    ESampleDataStorageDepth = 11,
};

struct SampleDataStorage{

    SampleData samples[ESampleDataStorageDepth];
    uint32_t nSamples;
    ZeroValueSampleData zvSamples[ESampleDataStorageDepth];
    uint32_t nZVSamples;
};

#include "SampleDataStorage_soa.h"

struct SampleDataStorageBuffer: public SOA<SampleDataStorage> {
private:
    SampleData* host_samples;
    uint32_t* host_nSamples;
    //ZeroValueSampleData* host_zvSamples {nullptr};
    //uint32_t* host_nZVSamples {nullptr};
public:
    uint32_t max_depth;
    SampleDataStorageBuffer() = default;
    SampleDataStorageBuffer(int n, openpgl::gpu::Device *device, bool managed = false) : SOA<SampleDataStorage>(n, device, managed), max_depth(ESampleDataStorageDepth) 
    {
        /* */
        if(!this->managed) {
            host_samples = new SampleData[(ESampleDataStorageDepth)*n];
            host_nSamples = new uint32_t[n];
            //host_zvSamples = new ZeroValueSampleData[(ESampleDataStorageDepth)*n];
            //host_nZVSamples = new uint32_t[n];
        } else {
            host_samples = nullptr;
            host_nSamples = nullptr;
            //host_zvSamples = nullptr;
            //host_nZVSamples = nullptr;
        }
        /* */
    }

    SampleDataStorageBuffer(std::string fileName, openpgl::gpu::Device *device)
    {
        std::filebuf fb;
        fb.open(fileName, std::ios::in | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);
        
        this->deserialize(is, device);
        fb.close();

        if(!this->managed) {
            host_samples = new SampleData[(ESampleDataStorageDepth)*this->nAlloc];
            host_nSamples = new uint32_t[this->nAlloc];
            //host_zvSamples = new ZeroValueSampleData[(ESampleDataStorageDepth)*n];
            //host_nZVSamples = new uint32_t[n];
        } else {
            host_samples = nullptr;
            host_nSamples = nullptr;
            //host_zvSamples = nullptr;
            //host_nZVSamples = nullptr;
        }
    }

/*
    ~SampleDataStorageBuffer()
    {
        if(!this->managed) {
            delete[] host_samples;
            delete[] host_nSamples;
            delete[] host_zvSamples;
            delete[] host_nZVSamples;
        }
    }
*/  
    SampleDataStorageBuffer(const SampleDataStorageBuffer& s){
        //SOA<SampleDataStorage>::operator(s);
        SOA<SampleDataStorage>::operator=(s);

        host_samples = s.host_samples;
        host_nSamples = s.host_nSamples;
        //host_zvSamples = s.host_zvSamples;
        //host_nZVSamples = s.host_nZVSamples;
    }

    SampleDataStorageBuffer &operator=(const SampleDataStorageBuffer& s){
        //SOA<SampleDataStorage>::operator(s);
        SOA<SampleDataStorage>::operator=(s);

        host_samples = s.host_samples;
        host_nSamples = s.host_nSamples;
        //host_zvSamples = s.host_zvSamples;
        //host_nZVSamples = s.host_nZVSamples;
        return *this;
    }
    
    OPENPGL_GPU_CALLABLE
    void AddSampleData(const int pixelIndex, const Point3& position, const Vector3& direction, const float pdf, const float distance, const Vector3 contribution, const bool volume) const
    {
        uint32_t nSample = nSamples[pixelIndex];
        //if(nSample < 10) 
        {
            uint32_t idx = nSample;
            uint32_t flags = 0;
            flags = volume ? flags | openpgl::cpp::SampleData::Flags::EInsideVolume : flags;
#ifndef USE_RAW_SAMPLE_DATA
            SampleData sd = {flags, position, direction, ((contribution[0] + contribution[1] + contribution[2]) / 3.f)/pdf, pdf, distance};
#else
            SampleData sd = {{position[0], position[1], position[2]}, {direction[0], direction[1], direction[2]}, ((contribution[0] + contribution[1] + contribution[2]) / 3.f)/pdf, pdf, distance, flags};
#endif
            samples[idx][pixelIndex] = sd;
            nSamples[pixelIndex] = nSample + 1;
            
        }
    }

    OPENPGL_GPU_CALLABLE
    void AddZeroValueSampleData(const int pixelIndex, const Point3& position, const Vector3& direction, const bool volume) const {
        uint32_t nZVSample = nZVSamples[pixelIndex];
        //if(nZVSample < 10) 
        {
            uint32_t idx = nZVSample;
#ifndef USE_RAW_SAMPLE_DATA
            ZeroValueSampleData zvSD = {volume, position, direction};
#else
            ZeroValueSampleData zvSD = {{position[0], position[1], position[2]}, {direction[0], direction[1], direction[2]}, volume};
#endif
            zvSamples[idx][pixelIndex] = zvSD;
            nZVSamples[pixelIndex] = nZVSample + 1;
        }
    }

    OPENPGL_GPU_CALLABLE
    inline void Reset(const int pixelIndex) const {
        nSamples[pixelIndex] = 0;
        nZVSamples[pixelIndex] = 0;
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


//#if !defined(__CUDACC__)
    void CollectSampleData(openpgl::cpp::SampleStorage& sampleStorage){
        int maxQueueSize = this->nAlloc;
        Timer timerCopy;
        if (!this->managed) {
            device->memcpyArrayFromGPU<uint32_t>(nSamples, host_nSamples, maxQueueSize);
            //device->wait();
            //device->memcpyArrayFromGPU<uint32_t>(nZVSamples, host_nZVSamples, maxQueueSize);
            //device->wait();
            for (size_t i = 0; i < ESampleDataStorageDepth; i++) {
                uint32_t idx = i*maxQueueSize;
                device->memcpyArrayFromGPU<SampleData>(samples[i], &(host_samples[idx]), maxQueueSize);
                //device->wait();
                //device->memcpyArrayFromGPU<ZeroValueSampleData>(zvSamples[i], &(host_zvSamples[idx]), maxQueueSize); 
                //device->wait();  
            }
            device->wait();
            std::cout << std::endl << "Copy: time =" << timerCopy.elapsed() * 1e-6 << " sec" << std::endl;
        }
    
        
        Timer timerPropagate;
        //ParallelFor(0, maxQueueSize, [&](int pixelIndex)
#if !defined(__CUDACC__)
        tbb::parallel_for(tbb::blocked_range<int>(0, maxQueueSize), [&](tbb::blocked_range<int> r)
        {
            for (int pixelIndex = r.begin(); pixelIndex < r.end(); pixelIndex++) {
#else
            
            //#pragma omp parallel for
            #pragma omp parallel num_threads(36)
            {
            #pragma omp for schedule(static,1024)
            for (int pixelIndex = 0; pixelIndex < maxQueueSize; pixelIndex++) {
#endif
                int nSamples = !managed ? host_nSamples[pixelIndex] : this->nSamples[pixelIndex];
                //if (nSamples > 0)
                //    std::cout << "nSamples = " << nSamples << std::endl;
                for (int n = 0; n < nSamples; n++) {
                    uint32_t idx = (n*maxQueueSize) + pixelIndex;
                    SampleData sd = !managed ? host_samples[idx] : this->samples[n][pixelIndex];
#ifndef USE_RAW_SAMPLE_DATA
                    openpgl::cpp::SampleData pg_sd;
                    pg_sd.position = {sd.position[0], sd.position[1], sd.position[2]};
                    pg_sd.direction ={sd.direction[0], sd.direction[1], sd.direction[2]};
                    pg_sd.weight = sd.weight;
                    pg_sd.pdf = sd.pdf;
                    pg_sd.distance = sd.distance;
                    pg_sd.flags = sd.flags;
                    
                    sampleStorage.AddSample(pg_sd);
#else
                    sampleStorage.AddSample(sd);
#endif
                }
/*
                uint32_t nZeroValueSamples = !managed ? host_nZVSamples[pixelIndex] : this->nZVSamples[pixelIndex];
                for (int n = 0; n < nZeroValueSamples; n++) {
                    //uint32_t idx = (n*this->nAlloc) + n;
                    uint32_t idx = (n*maxQueueSize) + pixelIndex;
                    ZeroValueSampleData zvsd = !managed ? host_zvSamples[idx]: this->zvSamples[n][pixelIndex];
                    openpgl::cpp::ZeroValueSampleData pg_zvsd;
                    pg_zvsd.position = {zvsd.position[0], zvsd.position[1], zvsd.position[2]};
                    pg_zvsd.direction = {zvsd.direction[0], zvsd.direction[1], zvsd.direction[2]};
                    pg_zvsd.volume = zvsd.volume;
                    sampleStorage.AddZeroValueSample(pg_zvsd); 
                }
*/
            }
#if !defined(__CUDACC__)
        }
        );
#else
        }
#endif
        std::cout << std::endl << "CollectSampleData: time(sec) = " << timerPropagate.elapsed() * 1e-6 << std::endl;
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