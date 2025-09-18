// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#ifndef KERNEL_FUNCTION
#define KERNEL_FUNCTION
#endif

namespace openpgl
{
struct SampleStatistics
{
    Point3 mean{0.0f};
    Vector3 variance{0.0f};
    float numSamples{0};
    float numZeroValueSamples{0.0f};

    BBox sampleBounds{openpgl::Vector3(std::numeric_limits<float>::max()), openpgl::Vector3(-std::numeric_limits<float>::max())};

    KERNEL_FUNCTION inline void clear()
    {
        mean = Point3(0.0f);
        variance = Vector3(0.0f);
        numSamples = 0.0f;
        numZeroValueSamples = 0.0f;
        sampleBounds.lower = openpgl::Vector3(std::numeric_limits<float>::max());
        sampleBounds.upper = openpgl::Vector3(-std::numeric_limits<float>::max());
    }

    KERNEL_FUNCTION inline void addSample(const Point3 sample)
    {
        numSamples++;
        float incWeight = 1.f / float(numSamples);
        OPENPGL_ASSERT(numSamples > 0.0f);
        OPENPGL_ASSERT(embree::isvalid(incWeight));
        OPENPGL_ASSERT(incWeight >= 0.0f);

        const Point3 oldMean = mean;
        mean += (sample - oldMean) * incWeight;
        variance += (((sample - oldMean) * (sample - mean)) - variance) * incWeight;
        sampleBounds.extend(sample);
        OPENPGL_ASSERT(isValid());
    }

    KERNEL_FUNCTION inline Point3 getMean() const
    {
        return mean;
    }

    KERNEL_FUNCTION inline Vector3 getVariance() const
    {
        return variance;
    }

    KERNEL_FUNCTION inline BBox getSampleBounds() const
    {
        return sampleBounds;
    }

    KERNEL_FUNCTION inline Vector3 getSampleBoundsExtend() const
    {
        return sampleBounds.upper - sampleBounds.lower;
    }

    KERNEL_FUNCTION inline bool hasValidBoundRange() const
    {
        bool validBoundRange = false;
        Vector3 boundExtend = sampleBounds.upper - sampleBounds.lower;
        if (boundExtend.x + boundExtend.y + boundExtend.z > 0.f)
        {
            validBoundRange = true;
        }
        return validBoundRange;
    }

    KERNEL_FUNCTION inline void decay(const float &a)
    {
        OPENPGL_ASSERT(a >= 0.f);
        numSamples *= a;
        numZeroValueSamples *= a;
    }

    SHARED_FUNCTION inline float getNumSamples() const
    {
        return numSamples;
    }

    KERNEL_FUNCTION inline void addNumZeroValueSamples(const int numZeroValueSamples)
    {
        this->numZeroValueSamples += numZeroValueSamples;
    }

    KERNEL_FUNCTION inline float getNumZeroValueSamples() const
    {
        return numZeroValueSamples;
    }

    KERNEL_FUNCTION void split(const uint8_t &splitDim, const float &splitPos, const float &decay, const bool &splitLower)
    {
        OPENPGL_ASSERT(decay > 0.0f && decay <= 1.0f);

        if (numSamples > 0.f)
        {
            float newVariance = variance[splitDim];
            const float stdDerivation = newVariance > 0.f ? std::sqrt(newVariance) : 0.f;

            newVariance = newVariance - newVariance / 4.0f;
            variance[splitDim] = newVariance;
            if (splitLower)
            {
                sampleBounds.lower[splitDim] = std::max(splitPos, sampleBounds.lower[splitDim]);
                mean[splitDim] = std::min(sampleBounds.upper[splitDim], mean[splitDim] + stdDerivation / 2.0f);
                // TODO: there are rare ocasions where this can happen (boarder of the head scene)
                // find a way to handle these
                // OPENPGL_ASSERT(mean[splitDim] >= sampleBounds.lower[splitDim]);
                // mean[splitDim] += stdDerivation / 2.0f;
            }
            else
            {
                sampleBounds.upper[splitDim] = std::min(splitPos, sampleBounds.upper[splitDim]);
                mean[splitDim] = std::max(sampleBounds.lower[splitDim], mean[splitDim] - stdDerivation / 2.0f);
                // OPENPGL_ASSERT(mean[splitDim] <= sampleBounds.upper[splitDim]);
                // mean[splitDim] -= stdDerivation / 2.0f;
            }

            numSamples *= decay;
            numZeroValueSamples *= decay;
        }
    }

    KERNEL_FUNCTION void merge(const SampleStatistics &b)
    {
        if (numSamples + b.numSamples == 0)
        {
            return;
        }

        const Point3 meanA = mean;
        const Point3 meanB = b.mean;

        const Vector3 varianceA = variance;
        const Vector3 varianceB = b.variance;

        const float numSamplesA = numSamples;
        const float numSamplesB = b.numSamples;

        const float weightA = numSamplesA / (numSamplesA + numSamplesB);
        const float weightB = 1.0f - weightA;
        mean = meanA * weightA + meanB * weightB;
        numSamples += numSamplesB;

        // Simple version to calculate the variance of two merged distributions
        // variance = (weightA * (varianceA + meanA * meanA) + weightB * (varianceB + meanB * meanB)) - (mean * mean);
        // Numerical more stable version to calculate the variance of two merged distributions
        const Point3 meanDiffA = meanA - mean;
        const Point3 meanDiffB = meanB - mean;
        variance = (weightA * varianceA + weightB * varianceB) + (weightA * (meanDiffA * meanDiffA) + weightB * (meanDiffB * meanDiffB));
        variance.x = variance.x >= 0.f ? variance.x : 0.f;
        variance.y = variance.y >= 0.f ? variance.y : 0.f;
        variance.z = variance.z >= 0.f ? variance.z : 0.f;
        sampleBounds.extend(b.sampleBounds);

        numZeroValueSamples += b.numZeroValueSamples;

        OPENPGL_ASSERT(isValid());
    }

    KERNEL_FUNCTION inline bool isValid() const
    {
        bool valid = true;
        valid = valid && numSamples >= 0.0f;
        valid = valid && numZeroValueSamples >= 0.0f;
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(mean.x);
        valid = valid && embree::isvalid(mean.y);
        valid = valid && embree::isvalid(mean.z);
        OPENPGL_ASSERT(valid);

        valid = valid && std::isfinite(variance.x);
        valid = valid && std::isfinite(variance.y);
        valid = valid && std::isfinite(variance.z);
        OPENPGL_ASSERT(valid);

        valid = valid && variance.x >= 0.f;
        valid = valid && variance.y >= 0.f;
        valid = valid && variance.z >= 0.f;
        OPENPGL_ASSERT(valid);

        return valid;
    }

    KERNEL_FUNCTION SampleStatistics operator()(const SampleStatistics &a, const SampleStatistics &b) const
    {
        SampleStatistics merged = a;
        merged.merge(b);
        return merged;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss.precision(15);
        ss << "SampleStatistics:" << std::endl;
        ss << "numSamples: " << numSamples << std::endl;
        ss << "numZeroValueSamples: " << numZeroValueSamples << std::endl;
        ss << "mean: " << mean[0] << ",\t" << mean[1] << ",\t" << mean[2] << std::endl;
        ss << "variance: " << variance[0] << ",\t" << variance[1] << ",\t" << variance[2] << std::endl;
        ss << "sampleBounds: [" << sampleBounds.lower[0] << ",\t" << sampleBounds.lower[1] << ",\t" << sampleBounds.lower[2] << "] \t [" << sampleBounds.upper[0] << ",\t"
           << sampleBounds.upper[1] << ",\t" << sampleBounds.upper[2] << "] " << std::endl;
        return ss.str();
    }

    KERNEL_FUNCTION void serialize(std::ostream &stream) const
    {
        stream.write(reinterpret_cast<const char *>(&mean), sizeof(Point3));
        stream.write(reinterpret_cast<const char *>(&variance), sizeof(Vector3));
        stream.write(reinterpret_cast<const char *>(&numSamples), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&numZeroValueSamples), sizeof(float));
        stream.write(reinterpret_cast<const char *>(&sampleBounds), sizeof(BBox));
    }

    KERNEL_FUNCTION void deserialize(std::istream &stream)
    {
        stream.read(reinterpret_cast<char *>(&mean), sizeof(Point3));
        stream.read(reinterpret_cast<char *>(&variance), sizeof(Vector3));
        stream.read(reinterpret_cast<char *>(&numSamples), sizeof(float));
        stream.read(reinterpret_cast<char *>(&numZeroValueSamples), sizeof(float));
        stream.read(reinterpret_cast<char *>(&sampleBounds), sizeof(BBox));
    }

    KERNEL_FUNCTION bool operator==(const SampleStatistics &b) const
    {
        bool equal = true;
        if (mean.x != b.mean.x || mean.y != b.mean.y || mean.z != b.mean.z || variance.x != b.variance.x || variance.y != b.variance.y || variance.z != b.variance.z ||
            numSamples != b.numSamples || sampleBounds.lower.x != b.sampleBounds.lower.x || sampleBounds.lower.y != b.sampleBounds.lower.y ||
            sampleBounds.lower.z != b.sampleBounds.lower.z || sampleBounds.upper.x != b.sampleBounds.upper.x || sampleBounds.upper.y != b.sampleBounds.upper.y ||
            sampleBounds.upper.z != b.sampleBounds.upper.z)
        {
            equal = false;
        }
        return equal;
    }
};

// 2^18 number of bins
#define INTEGER_BINS float(1 << 18)
#define INTEGER_SAMPLE_STATS_BOUND_SCALE (1.0f + 2.f / INTEGER_BINS)

struct IntegerSampleStatistics
{
    Point3i mean{0};
    Vector3i variance{0};
    uint32_t numSamples{0};
    // measured sample bound in the discretized integer domain
    BBoxi intSampleBounds{openpgl::Vector3i(std::numeric_limits<int>::max()), openpgl::Vector3i(-std::numeric_limits<int>::max())};
    // actual measured sample bound (float)
    Vector3 sampleBoundsMin{0};
    Vector3 sampleBoundsMax{0};

    // sample bound stats to center the collected samples before discretization
    Vector3 sampleBoundsCenter{0};
    Vector3 sampleBoundsHalfExtend{0};
    Vector3 invSampleBoundsHalfExtend{0};

    std::string toString() const {
        std::stringstream ss;
        ss.precision(15);
        ss << "IntegerSampleStatistics:" << std::endl;
        ss << "numSamples: " << numSamples << std::endl;
        //ss << "numZeroValueSamples: " << numZeroValueSamples << std::endl;
        ss << "mean: " << mean[0] << ",\t" << mean[1] << ",\t" << mean[2] << std::endl;
        ss << "variance: " << variance[0] << ",\t" << variance[1] << ",\t" << variance[2] << std::endl;
        ss << "intSampleBounds: [" << intSampleBounds.lower[0] << ",\t" << intSampleBounds.lower[1] << ",\t" << intSampleBounds.lower[2] << "] \t [" << intSampleBounds.upper[0] << ",\t"
            << intSampleBounds.upper[1] << ",\t" << intSampleBounds.upper[2] << "] " << std::endl;


        ss << "scaledBounds: [" << sampleBoundsMin[0] << ",\t" << sampleBoundsMin[1] << ",\t" << sampleBoundsMin[2] << "] \t [" << sampleBoundsMax[0] << ",\t"
            << sampleBoundsMax[1] << ",\t" << sampleBoundsMax[2] << "] " << std::endl;
        ss << "center: " << sampleBoundsCenter[0] << ",\t" << sampleBoundsCenter[1] << ",\t" << sampleBoundsCenter[2] << std::endl;
        ss << "halfExtend: " << sampleBoundsHalfExtend[0] << ",\t" << sampleBoundsHalfExtend[1] << ",\t" << sampleBoundsHalfExtend[2] << std::endl;
        ss << "invHalfExtend: " << invSampleBoundsHalfExtend[0] << ",\t" << invSampleBoundsHalfExtend[1] << ",\t" << invSampleBoundsHalfExtend[2] << std::endl;
        return ss.str();
    }

    KERNEL_FUNCTION IntegerSampleStatistics()
    {
        mean = Point3i(0);
        variance = Vector3i(0);
        numSamples = 0;
        intSampleBounds = BBoxi(openpgl::Vector3i(std::numeric_limits<int>::max()), openpgl::Vector3i(-std::numeric_limits<int>::max()));
        sampleBoundsMin = Vector3(0);
        sampleBoundsMax = Vector3(0);
        sampleBoundsCenter = Vector3(0);
        sampleBoundsHalfExtend = Vector3(0);
        invSampleBoundsHalfExtend = Vector3(0);
    }

    KERNEL_FUNCTION IntegerSampleStatistics(const BBox &bounds)
    {
        mean = Point3i(0);
        variance = Vector3i(0);
        numSamples = 0;
        intSampleBounds = BBoxi(openpgl::Vector3i(std::numeric_limits<int>::max()), openpgl::Vector3i(-std::numeric_limits<int>::max()));

        // scaling the boundary of the samples to avoid discretization problems at the boundaries
        BBox scaledBounds = bounds;
        Vector3 center = scaledBounds.center();
        scaledBounds.lower = center + INTEGER_SAMPLE_STATS_BOUND_SCALE * (scaledBounds.lower - center);
        scaledBounds.upper = center + INTEGER_SAMPLE_STATS_BOUND_SCALE * (scaledBounds.upper - center);

        sampleBoundsMin = scaledBounds.lower;
        sampleBoundsMax = scaledBounds.upper;
        sampleBoundsHalfExtend = (scaledBounds.upper - scaledBounds.lower) * 0.5f;

        // Checking and compenstaing for sampled bounds with dimensions of zero extend (e.g. plane)
        invSampleBoundsHalfExtend.x = sampleBoundsHalfExtend.x > 0.f ? 1.0f / sampleBoundsHalfExtend.x : 0.f;
        invSampleBoundsHalfExtend.y = sampleBoundsHalfExtend.y > 0.f ? 1.0f / sampleBoundsHalfExtend.y : 0.f;
        invSampleBoundsHalfExtend.z = sampleBoundsHalfExtend.z > 0.f ? 1.0f / sampleBoundsHalfExtend.z : 0.f;
        sampleBoundsCenter = sampleBoundsMin + sampleBoundsHalfExtend;

        OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.x));
        OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.y));
        OPENPGL_ASSERT(embree::isvalid(invSampleBoundsHalfExtend.z));
    }

    KERNEL_FUNCTION inline void addSample(const Point3 sample)
    {
        numSamples++;
        Point3 tmpSample = ((sample - sampleBoundsCenter) * invSampleBoundsHalfExtend);
        Vector3 tmpVariance = (tmpSample * tmpSample) * INTEGER_BINS;
        tmpSample *= INTEGER_BINS;

        OPENPGL_ASSERT(embree::isvalid(tmpSample.x));
        OPENPGL_ASSERT(embree::isvalid(tmpSample.y));
        OPENPGL_ASSERT(embree::isvalid(tmpSample.z));

        Point3i iSample(tmpSample.x, tmpSample.y, tmpSample.z);
        mean += iSample;
        variance += Vector3i(tmpVariance.x, tmpVariance.y, tmpVariance.z);

        intSampleBounds.extend(iSample);
        OPENPGL_ASSERT(isValid());
    }

    KERNEL_FUNCTION void merge(const IntegerSampleStatistics &b)
    {
        if (numSamples + b.numSamples == 0)
        {
            return;
        }
        mean += b.mean;
        variance += b.variance;
        numSamples += b.numSamples;
        intSampleBounds.extend(b.intSampleBounds);
        OPENPGL_ASSERT(isValid());
    }

    KERNEL_FUNCTION static IntegerSampleStatistics merge(const IntegerSampleStatistics &a, const IntegerSampleStatistics &b)
    {
        if (a.numSamples + b.numSamples == 0)
        {
            return a;
        }
        IntegerSampleStatistics stats = a;
        stats.mean += b.mean;
        stats.variance += b.variance;
        stats.numSamples += b.numSamples;
        stats.intSampleBounds.extend(b.intSampleBounds);
        OPENPGL_ASSERT(stats.isValid());
        return stats;
    }

    KERNEL_FUNCTION inline bool isValid() const
    {
        bool valid = true;
        valid = valid && numSamples >= 0.0f;
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(mean.x);
        valid = valid && embree::isvalid(mean.y);
        valid = valid && embree::isvalid(mean.z);
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(variance.x);
        valid = valid && embree::isvalid(variance.y);
        valid = valid && embree::isvalid(variance.z);
        OPENPGL_ASSERT(valid);

        valid = valid && variance.x >= 0;
        valid = valid && variance.y >= 0;
        valid = valid && variance.z >= 0;
        OPENPGL_ASSERT(valid);

        return valid;
    }

    KERNEL_FUNCTION SampleStatistics getSampleStatistics() const
    {
        SampleStatistics sampleStats;
        if (numSamples > 0)
        {
            auto unpack = [&](const Vector3i &v) {
                return sampleBoundsCenter+ (Vector3(v[0], v[1], v[2]) / INTEGER_BINS) * sampleBoundsHalfExtend;
            };
            
            Vector3 lowerCollectedSampleBound = unpack(intSampleBounds.lower);
            Vector3 upperCollectedSampleBound = unpack(intSampleBounds.upper);
            Vector3 collectedSampleBoundExtend = (upperCollectedSampleBound - lowerCollectedSampleBound);
            Vector3 halfCollectedSampleBoundExtend = (upperCollectedSampleBound - lowerCollectedSampleBound) * 0.5f;
            Vector3 halfBinSize = Vector3(0.5f / INTEGER_BINS) * sampleBoundsHalfExtend;

            float invNumSamples = 1.f / float(numSamples);
            Point3 sampleMean = (Point3(mean.x, mean.y, mean.z) * invNumSamples);
            sampleMean /= INTEGER_BINS;
            Point3 sampleMeanBin = sampleMean;
            sampleMean = sampleMean * sampleBoundsHalfExtend;
            sampleMean += sampleBoundsCenter;

            // Ensuring that them sample mean position is inside the collected/measured sample bounds
            sampleMean.x = sampleMean.x <= lowerCollectedSampleBound.x ? lowerCollectedSampleBound.x + std::min(halfCollectedSampleBoundExtend.x, halfBinSize.x) : sampleMean.x;
            sampleMean.y = sampleMean.y <= lowerCollectedSampleBound.y ? lowerCollectedSampleBound.y + std::min(halfCollectedSampleBoundExtend.y, halfBinSize.y) : sampleMean.y;
            sampleMean.z = sampleMean.z <= lowerCollectedSampleBound.z ? lowerCollectedSampleBound.z + std::min(halfCollectedSampleBoundExtend.z, halfBinSize.z) : sampleMean.z;
            sampleMean.x = sampleMean.x >= upperCollectedSampleBound.x ? upperCollectedSampleBound.x - std::min(halfCollectedSampleBoundExtend.x, halfBinSize.x) : sampleMean.x;
            sampleMean.y = sampleMean.y >= upperCollectedSampleBound.y ? upperCollectedSampleBound.y - std::min(halfCollectedSampleBoundExtend.y, halfBinSize.y) : sampleMean.y;
            sampleMean.z = sampleMean.z >= upperCollectedSampleBound.z ? upperCollectedSampleBound.z - std::min(halfCollectedSampleBoundExtend.z, halfBinSize.z) : sampleMean.z;

            Vector3 sampleVariance = (Vector3(variance.x, variance.y, variance.z) / (INTEGER_BINS)) * invNumSamples;
            sampleVariance -= sampleMeanBin * sampleMeanBin;
            sampleVariance = Vector3(std::max(0.f, sampleVariance.x), std::max(0.f, sampleVariance.y), std::max(0.f, sampleVariance.z));
            // sampleVariance = Vector3(std::fabs(sampleVariance.x), std::fabs(sampleVariance.y), std::fabs(sampleVariance.z));

            sampleVariance = sampleVariance * (sampleBoundsHalfExtend * sampleBoundsHalfExtend);
            // Ensuring that the estimated variance is in the right bounds.
            sampleVariance.x = std::min(collectedSampleBoundExtend.x * collectedSampleBoundExtend.x, sampleVariance.x);
            sampleVariance.y = std::min(collectedSampleBoundExtend.y * collectedSampleBoundExtend.y, sampleVariance.y);
            sampleVariance.z = std::min(collectedSampleBoundExtend.z * collectedSampleBoundExtend.z, sampleVariance.z);

            sampleStats.mean = sampleMean;
            sampleStats.numSamples = numSamples;
            sampleStats.variance = sampleVariance;

            // setting the variance to zero if the measured integer sample bound is zero
            sampleStats.variance.x = intSampleBounds.upper.x - intSampleBounds.lower.x <= 0 ? 0.f : sampleStats.variance.x;
            sampleStats.variance.y = intSampleBounds.upper.y - intSampleBounds.lower.y <= 0 ? 0.f : sampleStats.variance.y;
            sampleStats.variance.z = intSampleBounds.upper.z - intSampleBounds.lower.z <= 0 ? 0.f : sampleStats.variance.z;

            // using the real (float) measured sample bound and not a transformed version of the integer sample bound for accuracy reasons
            sampleStats.sampleBounds.lower = lowerCollectedSampleBound;
            sampleStats.sampleBounds.upper = upperCollectedSampleBound;
        }
        OPENPGL_ASSERT(sampleStats.isValid());
        return sampleStats;
    }
};

}  // namespace openpgl