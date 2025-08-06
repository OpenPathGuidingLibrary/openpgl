#if defined(OPENPGL_GPU_SYCL)
//#define expf(x) ::sycl::native::exp(((float)(x)))
#if defined(OPENPGL_GPU_SYCL)
#ifdef expf
    #undef expf
#endif
#ifdef logf
    #undef logf
#endif
#endif
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846264338327950288f
#endif

OPENPGL_GPU_CALLABLE inline Vector3 sphericalDirection(const float &cosTheta, const float &sinTheta, const float &cosPhi, const float &sinPhi)
{
    return Vector3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
};

OPENPGL_GPU_CALLABLE inline Vector3 sphericalDirection(const float &theta, const float &phi)
{
    const float cosTheta = std::cos(theta);
    const float sinTheta = std::sin(theta);
    const float cosPhi = std::cos(phi);
    const float sinPhi = std::sin(phi);

    return sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
};

OPENPGL_GPU_CALLABLE inline Vector3 squareToUniformSphere(const pgl_vec2f sample)
{
    float z = 1.0f - 2.0f * sample.y;
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
    float r = std::sqrt(std::max(0.f, (1.0f - z * z)));
#else
    float r = std::sqrt(std::fmaxf(0.f, (1.0f - z * z)));
#endif
    float sinPhi, cosPhi;
    pgl_sincosf(2.0f * float(M_PIf) * sample.x, &sinPhi, &cosPhi);
    return Vector3(r * cosPhi, r * sinPhi, z);
}

OPENPGL_GPU_CALLABLE inline pgl_vec2f directionToCanonical(const Vector3 &direction)
{
    // if (!std::isfinite(direction[0]) || !std::isfinite(direction[1]) || !std::isfinite(direction[2])) {
    //     return {0.f, 0.f};
    // }
#if !defined(OPENPGL_GPU_CUDA)
    const float cosTheta = std::min(std::max(direction[2], -1.f), 1.f);
#else
    const float cosTheta = std::fminf(std::fmaxf(direction[2], -1.f), 1.f);
#endif
    float phi = std::atan2(direction[1], direction[0]);
    while (phi < 0)
        phi += 2.f * M_PIf;

    return {(cosTheta + 1.f) / 2.f, phi / (2.f * M_PIf)};
}

OPENPGL_GPU_CALLABLE inline float kappaToMeanCosine(const float kappa)
{
    return kappa > 0.f ? (1.f / std::tanh(kappa)) - (1.f / kappa) : 0.f;
}

OPENPGL_GPU_CALLABLE inline float meanCosineToKappa(const float meanCosine)
{
    const float meanCosine2 = meanCosine * meanCosine;
    const float dim = 3.f;
    return meanCosine < 1.f ? (meanCosine * dim - meanCosine * meanCosine2) / (1.f - meanCosine2) : 0.f;
}

OPENPGL_GPU_CALLABLE inline float convolvePDF(const Vector3 &meanDirection, const Vector3 &direction, const float meanCosine0, const float meanCosine1)
{
    const float meanCosine = meanCosine0 * meanCosine1;
    const float kappa = meanCosineToKappa(meanCosine);
    const float norm = kappa > 0.f ? kappa / (2.f * M_PIf * (1.f - expf(-2.f * kappa))) : ONE_OVER_FOUR_PI;
    const float cosTheta = direction[0] * meanDirection[0] + direction[1] * meanDirection[1] + direction[2] * meanDirection[2];
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
    const float costThetaMinusOne = std::min(cosTheta - 1.f, 0.f);
#else
    const float costThetaMinusOne = std::fminf(cosTheta - 1.f, 0.f);
#endif
    const float eval = norm * expf(kappa * costThetaMinusOne);
    return eval;
};

OPENPGL_GPU_CALLABLE inline Vector3 toVector3(const float vec[3]) {
    return {vec[0], vec[1], vec[2]};
}

OPENPGL_GPU_CALLABLE inline Vector3 toVector3(const pgl_vec3f &vec) {
    return {vec.x, vec.y, vec.z};
}

OPENPGL_GPU_CALLABLE inline pgl_vec3f toVec3f(const Vector3 &vec) {
    return {vec.vec.x, vec.vec.y, vec.vec.z};
}


OPENPGL_GPU_CALLABLE inline float getNormalization(const float kappa) {
    if (kappa > 0.0f)
        return kappa / (2.0f * M_PIf * (1.0f - expf(-2.0f * kappa)));
    else
        return 1.f / (4.f * M_PIf);
}

struct VMF {
    float weight;
    Vector3 meanDirection;
    float kappa;
};

OPENPGL_GPU_CALLABLE VMF inline productVMF(const VMF &a, const VMF &b) {
    VMF p;
    p.meanDirection = a.meanDirection * a.kappa + b.meanDirection * b.kappa;
    p.kappa = std::sqrt(dot(p.meanDirection, p.meanDirection));
    if (p.kappa < 1e-3f) {
        p.meanDirection = a.meanDirection;
        p.kappa = 0;
    } else {
        p.meanDirection /= p.kappa;
    }
    const float normA = getNormalization(a.kappa);
    const float normB = getNormalization(b.kappa);
    const float normP = getNormalization(p.kappa);
    const float cosThetaA = dot(a.meanDirection, p.meanDirection);
    const float cosThetaB = dot(b.meanDirection, p.meanDirection);
    p.weight = a.weight * b.weight *
               normA * normB / normP *
               expf(a.kappa * (cosThetaA - 1) + b.kappa * (cosThetaB - 1));
    return p;
}

OPENPGL_GPU_CALLABLE VMF inline getVMFCosine(const Vector3 &normal) {
    VMF v;
    v.weight = 1;
    v.meanDirection = normal;
    v.kappa = 2.18853f;
    return v;
}

OPENPGL_GPU_CALLABLE VMF inline getVMFPFRep(const VMMPhaseFunctionRepresentation &pfRep, const int k, const Vector3 &dir, const float meanCosine) {
    VMF v;
    v.weight = pfRep.weights[k];
    v.meanDirection = (meanCosine * pfRep.meanCosines[k]) > 0.f ? dir : dir * -1.f;
    v.kappa = pfRep.kappas[k];
    return v;
}

OPENPGL_GPU_CALLABLE inline Vector3 sampleVMF(const Vector3 &meanDirection, const float kappa, const pgl_vec2f &sample) {
    pgl_vec2f _sample = sample;

    const float eMinus2Kappa = expf(-2.0f * kappa);
    Vector3 sampledDirection;
    if (kappa == 0.0f)
    {
        sampledDirection = squareToUniformSphere(_sample);
    }
    else
    {
        float cosTheta = 1.f + logf(1.0f + ((eMinus2Kappa - 1.f) * _sample.x)) / kappa;

// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
        // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
        cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));
#else
        cosTheta = std::fminf(1.0f, std::fmaxf(cosTheta, -1.f));
#endif
        const float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

        const float phi = 2.f * float(M_PIf) * _sample.y;

        float sinPhi, cosPhi;
        pgl_sincosf(phi, &sinPhi, &cosPhi);
        sampledDirection = sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
    }

    const Vector3 dx0(0.0f, meanDirection[2], -meanDirection[1]);
    const Vector3 dx1(-meanDirection[2], 0.0f, meanDirection[0]);
    const Vector3 dx = normalize(dot(dx0, dx0) > dot(dx1, dx1) ? dx0 : dx1);
    const Vector3 dy = normalize(cross(meanDirection, dx));

    Vector3 out = dx * sampledDirection[0] + dy * sampledDirection[1] + meanDirection * sampledDirection[2];
    return {out[0], out[1], out[2]};
}

OPENPGL_GPU_CALLABLE inline float pdfVMF(const Vector3 &meanDirection, const float kappa, const Vector3 &dir) {
    float norm = kappa > 0.f ? kappa / (2.f * M_PIf * (1.f - expf(-2.f * kappa))) : ONE_OVER_FOUR_PI;
    const float cosThetaK = dir[0] * meanDirection[0] + dir[1] * meanDirection[1] + dir[2] * meanDirection[2];
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
    const float costThetaMinusOneK = std::min(cosThetaK - 1.f, 0.f);
#else
    const float costThetaMinusOneK = std::fminf(cosThetaK - 1.f, 0.f);
#endif
    return norm * expf(kappa * costThetaMinusOneK);
}

template <int maxComponents>
struct ParallaxAwareVonMisesFisherMixture : public FlatVMM<maxComponents>
{
   public:
    ParallaxAwareVonMisesFisherMixture() {}

   private:
    OPENPGL_GPU_CALLABLE inline uint32_t selectComponent(float &sample) const
    {
        int selectedComponent{0};
        float searched = sample;
        float sumWeights = 0.0f;
        float cdf = 0.0f;

        while (true)
        {
            cdf = this->_weights[selectedComponent];
            if (sumWeights + cdf >= searched || selectedComponent + 1 >= this->_numComponents)
            {
                break;
            }
            else
            {
                sumWeights += cdf;
                selectedComponent++;
            }
        }
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
        sample = std::min(1.0f - FLT_EPSILON, (searched - sumWeights) / cdf);
#else
        sample = std::fminf(1.0f - FLT_EPSILON, (searched - sumWeights) / cdf);
#endif
        return selectedComponent;
    }

    OPENPGL_GPU_CALLABLE VMF getVMFParallax(const int i, const Vector3 &pos) const {
        VMF vmf;
        vmf.weight = this->_weights[i];
        vmf.meanDirection = normalize(
            toVector3(this->_pivotPosition) - pos +
            toVector3(this->_meanDirections[i]) * this->_distances[i]
        );
        vmf.kappa = this->_kappas[i];
        return vmf;
    }

    OPENPGL_GPU_CALLABLE inline std::pair<uint32_t, uint32_t> selectComponentProductPhase(
        const Vector3 &pos, const Vector3 &dir, const float meanCosine, const VMMPhaseFunctionRepresentation &pfRep, float &sample) const
    {
        // TODO https://diglib.eg.org/server/api/core/bitstreams/13f2db6c-bf62-4f45-88a1-4ca13275fee6/content
        float sumWeights = 0.f;
        int selI{0}, selK{0};
        for (int i = 0; i < this->_numComponents; i++) {
            const VMF a = getVMFParallax(i, pos);
            for (int k = 0; k < pfRep.K; k++) {
                const VMF vmf = productVMF(a, getVMFPFRep(pfRep, k, dir, meanCosine));

                sumWeights += vmf.weight;
                const float p = vmf.weight / sumWeights;
                if (sample <= p) {
                    selI = i;
                    selK = k;
                    sample = sample / p;
                } else {
                    sample = (sample - p) / (1 - p);
                }
                sample = std::clamp(sample, 0.0f, 1.0f - FLT_EPSILON);
            }
        }

        return {selI, selK};
    }

    OPENPGL_GPU_CALLABLE inline uint32_t selectComponentProduct(
        const Vector3 &pos, const Vector3 &normal, float &sample) const
    {
        // TODO https://diglib.eg.org/server/api/core/bitstreams/13f2db6c-bf62-4f45-88a1-4ca13275fee6/content
        float sumWeights = 0.f;
        int selI{0};
        for (int i = 0; i < this->_numComponents; i++) {
            VMF vmf = productVMF(getVMFParallax(i, pos), getVMFCosine(normal));
    
            sumWeights += vmf.weight;
            const float p = vmf.weight / sumWeights;
            if (sample <= p) {
                selI = i;
                sample = sample / p;
            } else {
                sample = (sample - p) / (1 - p);
            }
            sample = std::clamp(sample, 0.0f, 1.0f - FLT_EPSILON);
        }
    
        return selI;
    }

    public:
    OPENPGL_GPU_CALLABLE pgl_vec3f sample(const pgl_vec2f sample) const
    {
        uint32_t selectedComponent{0};
        // First, identify component we want to sample

        pgl_vec2f _sample = sample;
        selectedComponent = selectComponent(_sample.y);

        Vector3 sampledDirection = Vector3(0.f, 0.f, 1.f);
        // Second, sample selected component
        const float sKappa = this->_kappas[selectedComponent];
        const float sEMinus2Kappa = expf(-2.0f * sKappa);
        Vector3 meanDirection = Vector3(this->_meanDirections[selectedComponent][0], this->_meanDirections[selectedComponent][1], this->_meanDirections[selectedComponent][2]);

        return toVec3f(sampleVMF(meanDirection, sKappa, _sample));
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f samplePos(const pgl_vec3f pos, const pgl_vec2f sample) const
    {
        uint32_t selectedComponent{0};
        // First, identify component we want to sample
        pgl_vec2f _sample = sample;
        selectedComponent = selectComponent(_sample.y);

        Vector3 sampledDirection(0.f, 0.f, 1.f);
        // Second, sample selected component
        const float sKappa = this->_kappas[selectedComponent];
        const float sEMinus2Kappa = expf(-2.0f * sKappa);
        Vector3 meanDirection(this->_meanDirections[selectedComponent][0], this->_meanDirections[selectedComponent][1], this->_meanDirections[selectedComponent][2]);
        // parallax shift
        Vector3 _pos = {pos.x, pos.y, pos.z};
        const Vector3 relativePivotShift = {this->_pivotPosition[0] - _pos[0], this->_pivotPosition[1] - _pos[1], this->_pivotPosition[2] - _pos[2]};
        meanDirection *= this->_distances[selectedComponent];
        meanDirection += relativePivotShift;
        float flength = length(meanDirection);
        meanDirection /= flength;
        
        return toVec3f(sampleVMF(meanDirection, sKappa, _sample));
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f samplePosProductPhase(const pgl_vec3f pos, const pgl_vec3f dir, const float meanCosine, const pgl_vec2f sample) const
    {
        const VMMPhaseFunctionRepresentation &pfRep = vMMSingleLobeHenyeyGreensteinOracle.getPhaseFunctionRepresentation(meanCosine);

        uint32_t selectedComponent{0};
        // First, identify component we want to sample
        pgl_vec2f _sample = sample;
        auto [i, k] = selectComponentProductPhase(toVector3(pos), toVector3(dir), meanCosine, pfRep, _sample.y);
        const VMF a = getVMFParallax(i, toVector3(pos));
        const VMF vmf = productVMF(a, getVMFPFRep(pfRep, k, toVector3(dir), meanCosine));

        return toVec3f(sampleVMF(vmf.meanDirection, vmf.kappa, _sample));
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f samplePosProductCosine(const pgl_vec3f pos, const pgl_vec3f normal, const pgl_vec2f sample) const
    {
        uint32_t selectedComponent{0};
        // First, identify component we want to sample
        pgl_vec2f _sample = sample;
        auto i = selectComponentProduct(toVector3(pos), toVector3(normal), _sample.y);
        VMF vmf = productVMF(getVMFParallax(i, toVector3(pos)), getVMFCosine(toVector3(normal)));
        
        return toVec3f(sampleVMF(vmf.meanDirection, vmf.kappa, _sample));
    }

    OPENPGL_GPU_CALLABLE float pdf(const pgl_vec3f dir) const
    {
        const Vector3 _dir = {dir.x, dir.y, dir.z};
        float pdf{0.f};
        for (int k = 0; k < this->_numComponents; k++)
        {
            const Vector3 meanDirection = {this->_meanDirections[k][0], this->_meanDirections[k][1], this->_meanDirections[k][2]};
            const float kappaK = this->_kappas[k];
            float norm = kappaK > 0.f ? kappaK / (2.f * M_PIf * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
            const float cosThetaK = _dir[0] * meanDirection[0] + _dir[1] * meanDirection[1] + _dir[2] * meanDirection[2];
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
            const float costThetaMinusOneK = std::min(cosThetaK - 1.f, 0.f);
#else
            const float costThetaMinusOneK = std::fminf(cosThetaK - 1.f, 0.f);
#endif
            pdf += this->_weights[k] * norm * expf(kappaK * costThetaMinusOneK);
        }
        return pdf;
    }

    OPENPGL_GPU_CALLABLE float pdfPos(const pgl_vec3f pos, const pgl_vec3f dir) const
    {
        const Vector3 _dir = {dir.x, dir.y, dir.z};
        const Vector3 _pos = {pos.x, pos.y, pos.z};
        const Vector3 relativePivotShift = {this->_pivotPosition[0] - _pos[0], this->_pivotPosition[1] - _pos[1], this->_pivotPosition[2] - _pos[2]};

        float pdf{0.f};
        for (int k = 0; k < this->_numComponents; k++)
        {
            Vector3 meanDirection = {this->_meanDirections[k][0], this->_meanDirections[k][1], this->_meanDirections[k][2]};
            meanDirection *= this->_distances[k];
            meanDirection += relativePivotShift;
            float flength = length(meanDirection);
            meanDirection /= flength;

            pdf += this->_weights[k] * pdfVMF(meanDirection, this->_kappas[k], toVector3(dir));            
        }
        return pdf;
    }

    OPENPGL_GPU_CALLABLE float pdfPosProductPhase(const pgl_vec3f pos, const pgl_vec3f vdir, const float meanCosine, const pgl_vec3f dir) const
    {
        const VMMPhaseFunctionRepresentation &pfRep = vMMSingleLobeHenyeyGreensteinOracle.getPhaseFunctionRepresentation(meanCosine);
        
        float sum{0.f}, pdf{0.f};
        for (int i = 0; i < this->_numComponents; i++) {
            VMF a = getVMFParallax(i, toVector3(pos));
            for (int k = 0; k < pfRep.K; k++) {
                VMF vmf = productVMF(a, getVMFPFRep(pfRep, k, toVector3(vdir), meanCosine));
                sum += vmf.weight;
                pdf += vmf.weight * pdfVMF(vmf.meanDirection, vmf.kappa, toVector3(dir));
            }
        }
        return pdf / sum;
    }

    OPENPGL_GPU_CALLABLE float pdfPosProductCosine(const pgl_vec3f pos, const pgl_vec3f normal, const pgl_vec3f dir) const
    {
        VMF c = getVMFCosine(toVector3(normal));
        float sum{0.f}, pdf{0.f};
        for (int i = 0; i < this->_numComponents; i++) {
            VMF vmf = productVMF(getVMFParallax(i, toVector3(pos)), c);
            sum += vmf.weight;
            pdf += vmf.weight * pdfVMF(vmf.meanDirection, vmf.kappa, toVector3(dir));
        }
        return pdf / sum;
    }

#ifdef OPENPGL_EF_RADIANCE_CACHES
    OPENPGL_GPU_CALLABLE pgl_vec3f fluence() const
    {
        return {this->_fluenceRGB[0], this->_fluenceRGB[1], this->_fluenceRGB[2]};
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f incomingRadiance(const pgl_vec3f pos, const pgl_vec3f dir) const
    {
        const Vector3 _dir = {dir.x, dir.y, dir.z};
        const Vector3 _pos = {pos.x, pos.y, pos.z};
        const Vector3 relativePivotShift = {this->_pivotPosition[0] - _pos[0], this->_pivotPosition[1] - _pos[1], this->_pivotPosition[2] - _pos[2]};

        Vector3 incomingRadiance = {0.f, 0.f, 0.f};
        for (int k = 0; k < this->_numComponents; k++)
        {
            Vector3 meanDirection = {this->_meanDirections[k][0], this->_meanDirections[k][1], this->_meanDirections[k][2]};
            meanDirection *= this->_distances[k];
            meanDirection += relativePivotShift;
            float flength = length(meanDirection);
            meanDirection /= flength;

            const float kappaK = this->_kappas[k];
            float norm = kappaK > 0.f ? kappaK / (2.f * M_PIf * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
            const float cosThetaK = _dir[0] * meanDirection[0] + _dir[1] * meanDirection[1] + _dir[2] * meanDirection[2];
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
            const float costThetaMinusOneK = std::min(cosThetaK - 1.f, 0.f);
#else
            const float costThetaMinusOneK = std::fminf(cosThetaK - 1.f, 0.f);
#endif
            const float eval = norm * expf(kappaK * costThetaMinusOneK);
            incomingRadiance[0] += this->_fluenceRGBWeights[k][0] * eval;
            incomingRadiance[1] += this->_fluenceRGBWeights[k][1] * eval;
            incomingRadiance[2] += this->_fluenceRGBWeights[k][2] * eval;
        }
        return {incomingRadiance[0], incomingRadiance[1], incomingRadiance[2]};
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f irradiance(const pgl_vec3f pos, const pgl_vec3f normal) const
    {
        const Vector3 _normal = {normal.x, normal.y, normal.z};
        const Vector3 _pos = {pos.x, pos.y, pos.z};
        const Vector3 relativePivotShift = {this->_pivotPosition[0] - _pos[0], this->_pivotPosition[1] - _pos[1], this->_pivotPosition[2] - _pos[2]};

        // mean cosine for a cosine lobe
        const float meanCosineVMF = kappaToMeanCosine(2.18853f);
        Vector3 inscatteredRadiance = {0.f, 0.f, 0.f};
        for (int k = 0; k < this->_numComponents; k++)
        {
            Vector3 meanDirection = {this->_meanDirections[k][0], this->_meanDirections[k][1], this->_meanDirections[k][2]};
            meanDirection *= this->_distances[k];
            meanDirection += relativePivotShift;
            float flength = length(meanDirection);
            meanDirection /= flength;

            const float kappaK = this->_kappas[k];
            const float meanCosineVMFK = kappaToMeanCosine(kappaK);
            const float eval = convolvePDF(meanDirection, _normal, meanCosineVMFK, meanCosineVMF);

            inscatteredRadiance[0] += this->_fluenceRGBWeights[k][0] * eval;
            inscatteredRadiance[1] += this->_fluenceRGBWeights[k][1] * eval;
            inscatteredRadiance[2] += this->_fluenceRGBWeights[k][2] * eval;
        }
        return {inscatteredRadiance[0], inscatteredRadiance[1], inscatteredRadiance[2]};
    }

    OPENPGL_GPU_CALLABLE pgl_vec3f inscatteredRadiance(const pgl_vec3f pos, const pgl_vec3f dir, const VMMPhaseFunctionRepresentationData &vprd) const
    {
        const Vector3 _dir = {dir.x, dir.y, dir.z};
        const Vector3 _pos = {pos.x, pos.y, pos.z};
        const Vector3 relativePivotShift = {this->_pivotPosition[0] - _pos[0], this->_pivotPosition[1] - _pos[1], this->_pivotPosition[2] - _pos[2]};

        // TODO: lookup VMF mean cosine for HG mean cosine
        Vector3 inscatteredRadiance = {0.f, 0.f, 0.f};
        for (int k = 0; k < this->_numComponents; k++)
        {
            Vector3 meanDirection = {this->_meanDirections[k][0], this->_meanDirections[k][1], this->_meanDirections[k][2]};
            meanDirection *= this->_distances[k];
            meanDirection += relativePivotShift;
            float flength = length(meanDirection);
            meanDirection /= flength;

            const float kappaK = this->_kappas[k];
            const float meanCosineVMFK = kappaToMeanCosine(kappaK);

            for (int i = 0; i < 4; i++)
            {
                const float meanCosineI = vprd.meanCosines[i] >= 0.f ? vprd.meanCosines[i] : -vprd.meanCosines[i];
                const Vector3 directionI = vprd.meanCosines[i] >= 0.f ? _dir : Vector3(-_dir[0], -_dir[1], -_dir[2]);
                const float eval = convolvePDF(meanDirection, directionI, meanCosineVMFK, meanCosineI);
                inscatteredRadiance[0] += vprd.weights[i] * this->_fluenceRGBWeights[k][0] * eval;
                inscatteredRadiance[1] += vprd.weights[i] * this->_fluenceRGBWeights[k][1] * eval;
                inscatteredRadiance[2] += vprd.weights[i] * this->_fluenceRGBWeights[k][2] * eval;
            }
        }
        return {inscatteredRadiance[0], inscatteredRadiance[1], inscatteredRadiance[2]};
    }
#endif
};