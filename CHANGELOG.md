Version History
===============

## Open PGL 0.3.1

-   `Field`:
    - Added `Reset()` function to reset a guiding field (e.g., when the lighting or the scene 
        geometry changed)

-   `PathSegmentStorage`:
    - Fixed bug when using `AddSample()`

## Open PGL 0.3.0

-   Added CMake Superbuild script to build Open PGL, including all its dependencies.      
    The dependencies (e.g., TBB and Embree) are downloaded, built, and installed automatically. 

-   Added support for different SIMD optimizations (SSE, AVX2, AVX-512).
        The optimization type can be chosen when initializing the `Device`.

-   Added support for directional quadtrees for the directional representation.

-   `PathSegmentStorage`:
    -   Added debug function `CalculatePixelEstimate` to validate if the stored
        path segment information represents the sampling behavior of the render
        (i.e., the resulting RGB value should match the pixel value the renderer
        adds to the framebuffer)

-   `SurfaceSamplingDistribution`:
    -   Added support for guiding based on the product of a normal-oriented
        cosine lobe and the incident radiance distribution:
        `(ApplyCosineProduct)`
        This feature is only supported for VMM-based directional distributions.
        Support can be checked with `SupportsApplyCosineProduct()`.

-   `VolumeSamplingDistribution`:
    -   Added support for guiding based on the product of a single lobe
        HG phase function and the incident radiance distribution:
        `ApplySingleLobeHenyeyGreensteinProduct()`
        This feature is only supported for VMM-based directional distributions.
        Support can be checked with `SupportsApplySingleLobeHenyeyGreensteinProduct()`.


## Open PGL 0.1.0

-   Initial release of Open PGL
    Features:
    -   Incremental learning/updating of a 5D spatio-directional radiance field
        from radiance samples (see `Field`).
    -   Directional representation based on (parallax-aware) von Mises-Fisher mixtures.
    -   `PathSegmentStorage` is a utility class to help keep track of all path segment 
        information and generate radiance samples when a path/random walk is finished/terminated.
    -   Support for guided importance sampling of directions on surfaces (see `SurfaceSamplingDistribution`)
        and inside volumes (see `VolumeSamplingDistribution`)

-   Added C-API and C++-API headers
    -   C-API: `#include <openpgl/openpgl.h>`
    -   C++-API: `#include <openpgl/cpp/OpenPGL.h>` and the namespace `openpgl::cpp::`

    