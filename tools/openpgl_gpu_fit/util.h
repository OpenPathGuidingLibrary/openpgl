//#define OPENPGL_GPU_CUDA
//#include "openpgl/gpu/OpenPGLGPU.h"

namespace openpgl{
namespace gpu {
namespace cuda {

static void writeBoundingBoxes(const std::vector<BBox> &boxes, const std::string &output) {
    // Open a file stream for writing.
    std::ofstream obj_file(output);

    if (!obj_file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output << std::endl;
        return;
    }
    
    int vertex_offset = 0;
    for (int i = 0; i < boxes.size(); i++) {
        const auto& box = boxes[i];
        Vector3 vertices[8] = {
            {box.lower.x, box.lower.y, box.lower.z}, // 0
            {box.upper.x, box.lower.y, box.lower.z}, // 1
            {box.upper.x, box.upper.y, box.lower.z}, // 2
            {box.lower.x, box.upper.y, box.lower.z}, // 3
            {box.lower.x, box.lower.y, box.upper.z}, // 4
            {box.upper.x, box.lower.y, box.upper.z}, // 5
            {box.upper.x, box.upper.y, box.upper.z}, // 6
            {box.lower.x, box.upper.y, box.upper.z}  // 7
        };

        for (int j = 0; j < 8; ++j) {
            obj_file << "v " << vertices[j].x << " " << vertices[j].y << " " << vertices[j].z << "\n";
        }

        obj_file << "o Box " << i << "\n";

        obj_file << "f " << 1 + vertex_offset << " " << 2 + vertex_offset << " " << 3 + vertex_offset << " " << 4 + vertex_offset << "\n"; // Front face
        obj_file << "f " << 5 + vertex_offset << " " << 8 + vertex_offset << " " << 7 + vertex_offset << " " << 6 + vertex_offset << "\n"; // Back face
        obj_file << "f " << 1 + vertex_offset << " " << 5 + vertex_offset << " " << 6 + vertex_offset << " " << 2 + vertex_offset << "\n"; // Bottom face
        obj_file << "f " << 4 + vertex_offset << " " << 3 + vertex_offset << " " << 7 + vertex_offset << " " << 8 + vertex_offset << "\n"; // Top face
        obj_file << "f " << 1 + vertex_offset << " " << 4 + vertex_offset << " " << 8 + vertex_offset << " " << 5 + vertex_offset << "\n"; // Left face
        obj_file << "f " << 2 + vertex_offset << " " << 6 + vertex_offset << " " << 7 + vertex_offset << " " << 3 + vertex_offset << "\n"; // Right face

        vertex_offset += 8;
    }

    obj_file.close();
}

static void write_point_cloud_to_obj(const std::vector<Vector3>& points, const std::string& filename) {
    // Open a file stream for writing.
    std::ofstream obj_file(filename);

    if (!obj_file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
    }

    for (const auto& point : points) {
        obj_file << "v " << point.x << " " << point.y << " " << point.z << "\n";
    }

    obj_file.close();
}

}
}
}