#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/h_ploc/hploc.h"
#include "kernels/h_ploc/hploc_wide.h"
#include "kernels/kernels.h"
#include "kernels/structs/aabb.h"
#include "kernels/structs/bvh_node.h"
#include "kernels/structs/morton_code.h"
#include "kernels/structs/wide_bvh_node.h"
#include "utils/defines.h"
#include "utils/utils.h"

namespace
{
  void check_cuda(cudaError_t err, const char* expr)
  {
    if (err == cudaSuccess) return;

    std::ostringstream out;
    out << expr << " failed: " << cudaGetErrorString(err) << " (" << static_cast<int>(err) << ")";
    throw std::runtime_error(out.str());
  }

#define TEST_CUDA(expr) check_cuda((expr), #expr)

  void require(bool condition, const std::string& message)
  {
    if (!condition) throw std::runtime_error(message);
  }

  bool almost_equal(float lhs, float rhs, float eps = 1e-5f) { return std::fabs(lhs - rhs) <= eps; }

  bool same_aabb(const AABB& lhs, const AABB& rhs)
  {
    return almost_equal(lhs.min_x, rhs.min_x) && almost_equal(lhs.min_y, rhs.min_y) && almost_equal(lhs.min_z, rhs.min_z) &&
           almost_equal(lhs.max_x, rhs.max_x) && almost_equal(lhs.max_y, rhs.max_y) && almost_equal(lhs.max_z, rhs.max_z);
  }

  void require_aabb_eq(const AABB& actual, const AABB& expected, const std::string& label)
  {
    if (same_aabb(actual, expected)) return;

    std::ostringstream out;
    out << label << " AABB mismatch";
    throw std::runtime_error(out.str());
  }

  struct TestScene {
    std::vector<float> vertices;
    std::vector<unsigned int> faces;
    std::vector<AABB> primitive_aabbs;
    AABB scene_aabb;

    unsigned int n_faces() const { return static_cast<unsigned int>(primitive_aabbs.size()); }
  };

  void append_triangle(TestScene& scene, float base_x, float base_y, float base_z)
  {
    const unsigned int base_index = static_cast<unsigned int>(scene.vertices.size() / 3);
    const std::array<float, 9> triangle = {
        base_x,
        base_y,
        base_z,
        base_x + 0.2f,
        base_y,
        base_z,
        base_x,
        base_y + 0.2f,
        base_z,
    };

    scene.vertices.insert(scene.vertices.end(), triangle.begin(), triangle.end());
    scene.faces.insert(scene.faces.end(), {base_index, base_index + 1, base_index + 2});
    scene.primitive_aabbs.push_back({base_x, base_y, base_z, base_x + 0.2f, base_y + 0.2f, base_z});
  }

  TestScene make_test_scene()
  {
    TestScene scene;

    append_triangle(scene, 0.0f, 0.0f, 0.0f);
    append_triangle(scene, 1.0f, 0.0f, 0.0f);
    append_triangle(scene, 10.0f, 0.0f, 0.0f);
    append_triangle(scene, 13.0f, 0.0f, 0.0f);
    append_triangle(scene, 16.5f, 0.0f, 0.0f);

    scene.scene_aabb = scene.primitive_aabbs.front();
    for (size_t i = 1; i < scene.primitive_aabbs.size(); ++i) {
      scene.scene_aabb = AABB::union_of(scene.scene_aabb, scene.primitive_aabbs[i]);
    }

    return scene;
  }

  TestScene make_large_test_scene()
  {
    TestScene scene;

    for (unsigned int row = 0; row < 3; ++row) {
      for (unsigned int col = 0; col < 11; ++col) {
        const float x = static_cast<float>(col) * 1.7f + static_cast<float>(row) * 0.13f;
        const float y = static_cast<float>(row) * 2.3f + static_cast<float>(col % 2) * 0.17f;
        const float z = static_cast<float>((col + row) % 3) * 0.11f;
        append_triangle(scene, x, y, z);
      }
    }

    scene.scene_aabb = scene.primitive_aabbs.front();
    for (size_t i = 1; i < scene.primitive_aabbs.size(); ++i) {
      scene.scene_aabb = AABB::union_of(scene.scene_aabb, scene.primitive_aabbs[i]);
    }

    return scene;
  }

  std::vector<BVHNode> build_binary_bvh(cudaStream_t stream, const TestScene& scene)
  {
    const unsigned int n_faces = scene.n_faces();
    const unsigned int max_nodes = 2 * n_faces - 1;

    float* d_vertices = nullptr;
    unsigned int* d_faces = nullptr;
    BVHNode* d_nodes = nullptr;
    unsigned int* d_parents = nullptr;
    MortonCode* d_morton_codes = nullptr;
    MortonCode* d_morton_codes_sorted = nullptr;
    unsigned int* d_cluster_ids = nullptr;
    unsigned int* d_cluster_ids_sorted = nullptr;
    unsigned int* d_n_clusters = nullptr;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;

    TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_vertices), sizeof(float) * scene.vertices.size()));
    TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_faces), sizeof(unsigned int) * scene.faces.size()));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_nodes), sizeof(BVHNode) * max_nodes));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_parents), sizeof(unsigned int) * max_nodes));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_morton_codes), sizeof(MortonCode) * n_faces));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_morton_codes_sorted), sizeof(MortonCode) * n_faces));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_cluster_ids), sizeof(unsigned int) * max_nodes));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_cluster_ids_sorted), sizeof(unsigned int) * n_faces));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_n_clusters), sizeof(unsigned int)));
      TEST_CUDA(cub::DeviceRadixSort::SortPairs(
          nullptr,
          sort_temp_storage_bytes,
          d_morton_codes,
          d_morton_codes_sorted,
          d_cluster_ids,
          d_cluster_ids_sorted,
          static_cast<int>(n_faces),
          2,
          32,
          stream));
      TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_sort_temp_storage), sort_temp_storage_bytes));

    try {
      TEST_CUDA(cudaMemcpyAsync(d_vertices, scene.vertices.data(), sizeof(float) * scene.vertices.size(), cudaMemcpyHostToDevice, stream));
      TEST_CUDA(cudaMemcpyAsync(d_faces, scene.faces.data(), sizeof(unsigned int) * scene.faces.size(), cudaMemcpyHostToDevice, stream));

      TEST_CUDA(cudaMemsetAsync(d_parents, 0xff, sizeof(unsigned int) * max_nodes, stream));
      TEST_CUDA(cudaMemcpyAsync(d_n_clusters, &n_faces, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

      cuda::hploc::build_leaves_nodes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
          d_faces,
          n_faces,
          d_vertices,
          d_nodes);
      cuda::fill_indices(stream, d_cluster_ids, n_faces);
      cuda::compute_mortons_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
          scene.scene_aabb,
          d_faces,
          d_vertices,
          d_morton_codes,
          n_faces);
      TEST_CUDA(cub::DeviceRadixSort::SortPairs(
          d_sort_temp_storage,
          sort_temp_storage_bytes,
          d_morton_codes,
          d_morton_codes_sorted,
          d_cluster_ids,
          d_cluster_ids_sorted,
          static_cast<int>(n_faces),
          2,
          32,
          stream));
      constexpr size_t block_size = 128;
      cuda::hploc::build_kernel<<<div_ceil(n_faces, block_size), block_size, 0, stream>>>(
          d_parents,
          d_morton_codes_sorted,
          d_nodes,
          d_cluster_ids_sorted,
          d_n_clusters,
          n_faces);
      TEST_CUDA(cudaGetLastError());
      TEST_CUDA(cudaStreamSynchronize(stream));

      unsigned int n_nodes = 0;
      TEST_CUDA(cudaMemcpy(&n_nodes, d_n_clusters, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      require(n_nodes == max_nodes, "Unexpected number of binary BVH nodes");

      std::vector<BVHNode> nodes(n_nodes);
      TEST_CUDA(cudaMemcpy(nodes.data(), d_nodes, sizeof(BVHNode) * n_nodes, cudaMemcpyDeviceToHost));

      TEST_CUDA(cudaFree(d_vertices));
      TEST_CUDA(cudaFree(d_faces));
      TEST_CUDA(cudaFree(d_nodes));
      TEST_CUDA(cudaFree(d_parents));
      TEST_CUDA(cudaFree(d_morton_codes));
      TEST_CUDA(cudaFree(d_morton_codes_sorted));
      TEST_CUDA(cudaFree(d_cluster_ids));
      TEST_CUDA(cudaFree(d_cluster_ids_sorted));
      TEST_CUDA(cudaFree(d_n_clusters));
      TEST_CUDA(cudaFree(d_sort_temp_storage));
      return nodes;
    } catch (...) {
      cudaFree(d_vertices);
      cudaFree(d_faces);
      cudaFree(d_nodes);
      cudaFree(d_parents);
      cudaFree(d_morton_codes);
      cudaFree(d_morton_codes_sorted);
      cudaFree(d_cluster_ids);
      cudaFree(d_cluster_ids_sorted);
      cudaFree(d_n_clusters);
      cudaFree(d_sort_temp_storage);
      throw;
    }
  }

  template <unsigned int Arity>
  std::vector<WideBVHNode<Arity>> build_wide_bvh(cudaStream_t stream, const std::vector<BVHNode>& binary_nodes, unsigned int n_faces)
  {
    const unsigned int max_nodes = static_cast<unsigned int>(binary_nodes.size());

    BVHNode* d_binary_nodes = nullptr;
    WideBVHNode<Arity>* d_wide_nodes = nullptr;
    unsigned long long* d_tasks = nullptr;
    unsigned int* d_next_task = nullptr;
    unsigned int* d_next_wide_node = nullptr;

    TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_binary_nodes), sizeof(BVHNode) * max_nodes));
    TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_wide_nodes), sizeof(WideBVHNode<Arity>) * max_nodes));
    TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_tasks), sizeof(unsigned long long) * n_faces));
    TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_next_task), sizeof(unsigned int)));
    TEST_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_next_wide_node), sizeof(unsigned int)));

    try {
      TEST_CUDA(cudaMemcpyAsync(d_binary_nodes, binary_nodes.data(), sizeof(BVHNode) * max_nodes, cudaMemcpyHostToDevice, stream));

      cuda::hploc::convert_to_wide<Arity>(stream, d_binary_nodes, d_wide_nodes, d_tasks, d_next_task, d_next_wide_node, n_faces);
      TEST_CUDA(cudaGetLastError());
      TEST_CUDA(cudaStreamSynchronize(stream));

      unsigned int n_wide_nodes = 0;
      TEST_CUDA(cudaMemcpy(&n_wide_nodes, d_next_wide_node, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      require(n_wide_nodes > 0 && n_wide_nodes <= max_nodes, "Unexpected number of wide BVH nodes");

      std::vector<WideBVHNode<Arity>> wide_nodes(n_wide_nodes);
      TEST_CUDA(cudaMemcpy(wide_nodes.data(), d_wide_nodes, sizeof(WideBVHNode<Arity>) * n_wide_nodes, cudaMemcpyDeviceToHost));

      TEST_CUDA(cudaFree(d_binary_nodes));
      TEST_CUDA(cudaFree(d_wide_nodes));
      TEST_CUDA(cudaFree(d_tasks));
      TEST_CUDA(cudaFree(d_next_task));
      TEST_CUDA(cudaFree(d_next_wide_node));
      return wide_nodes;
    } catch (...) {
      cudaFree(d_binary_nodes);
      cudaFree(d_wide_nodes);
      cudaFree(d_tasks);
      cudaFree(d_next_task);
      cudaFree(d_next_wide_node);
      throw;
    }
  }

  void validate_binary_topology(const std::vector<BVHNode>& nodes, const TestScene& scene)
  {
    const unsigned int n_faces = scene.n_faces();
    require(nodes.size() == 2 * n_faces - 1, "Binary BVH must be a full tree");

    std::vector<bool> visited(nodes.size(), false);
    std::vector<unsigned int> primitives;

    auto visit = [&](auto&& self, unsigned int node_index) -> AABB {
      require(node_index < nodes.size(), "Binary node index is out of range");
      require(!visited[node_index], "Binary BVH contains a cycle or a shared child");

      visited[node_index] = true;
      const BVHNode& node = nodes[node_index];
      if (node.is_leaf()) {
        require(node.right_child_index < n_faces, "Leaf stores invalid primitive id");
        require_aabb_eq(node.aabb, scene.primitive_aabbs[node.right_child_index], "Leaf");
        primitives.push_back(node.right_child_index);
        return node.aabb;
      }

      require(node.right_child_index < nodes.size(), "Internal node stores invalid child index");
      const AABB left_aabb = self(self, node.left_child_index);
      const AABB right_aabb = self(self, node.right_child_index);
      require_aabb_eq(node.aabb, AABB::union_of(left_aabb, right_aabb), "Internal node");
      return node.aabb;
    };

    visit(visit, static_cast<unsigned int>(nodes.size() - 1));
    require(std::all_of(visited.begin(), visited.end(), [](bool flag) { return flag; }), "Binary BVH contains unreachable nodes");

    std::sort(primitives.begin(), primitives.end());
    require(primitives.size() == n_faces, "Binary BVH must contain all primitives exactly once");
    for (unsigned int primitive_id = 0; primitive_id < n_faces; ++primitive_id) {
      require(primitives[primitive_id] == primitive_id, "Binary BVH primitive set mismatch");
    }
  }

  template <unsigned int Arity>
  void validate_wide_topology(const std::vector<WideBVHNode<Arity>>& wide_nodes, const TestScene& scene)
  {
    const unsigned int n_faces = scene.n_faces();
    std::vector<bool> visited(wide_nodes.size(), false);
    std::vector<unsigned int> primitives;

    auto visit = [&](auto&& self, unsigned int node_index) -> AABB {
      require(node_index < wide_nodes.size(), "Wide BVH node index is out of range");
      require(!visited[node_index], "Wide BVH contains a cycle or a shared child");

      visited[node_index] = true;
      const WideBVHNode<Arity>& node = wide_nodes[node_index];

      AABB merged{};
      bool has_child = false;
      unsigned int expected_valid_mask = 0;
      for (unsigned int slot = 0; slot < Arity; ++slot) {
        const unsigned int bit = 1u << slot;
        if ((node.valid_mask & bit) == 0u) {
          require(node.child_indices[slot] == INVALID_INDEX, "Invalid wide slot must keep INVALID_INDEX");
          continue;
        }

        expected_valid_mask |= bit;
        const AABB child_aabb = [&]() {
          if ((node.primitive_mask & bit) != 0u) {
            require(node.child_indices[slot] < n_faces, "Wide leaf stores invalid primitive id");
            primitives.push_back(node.child_indices[slot]);
            return scene.primitive_aabbs[node.child_indices[slot]];
          }

          return self(self, node.child_indices[slot]);
        }();

        require_aabb_eq(node.child_aabbs[slot], child_aabb, "Wide child");
        merged = has_child ? AABB::union_of(merged, child_aabb) : child_aabb;
        has_child = true;
      }

      require(has_child, "Wide node must have at least one child");
      require(node.valid_mask == expected_valid_mask, "Wide valid mask must be contiguous from slot zero");
      require_aabb_eq(node.aabb, merged, "Wide node");
      return node.aabb;
    };

    visit(visit, 0);
    require(std::all_of(visited.begin(), visited.end(), [](bool flag) { return flag; }), "Wide BVH contains unreachable nodes");

    std::sort(primitives.begin(), primitives.end());
    require(primitives.size() == n_faces, "Wide BVH must contain all primitives exactly once");
    for (unsigned int primitive_id = 0; primitive_id < n_faces; ++primitive_id) {
      require(primitives[primitive_id] == primitive_id, "Wide BVH primitive set mismatch");
    }
  }

  void test_hploc_binary_topology(cudaStream_t stream)
  {
    const TestScene scene = make_test_scene();
    const std::vector<BVHNode> nodes = build_binary_bvh(stream, scene);

    require(nodes[0].is_leaf() && nodes[0].right_child_index == 0, "Leaf 0 mismatch");
    require(nodes[1].is_leaf() && nodes[1].right_child_index == 1, "Leaf 1 mismatch");
    require(nodes[2].is_leaf() && nodes[2].right_child_index == 2, "Leaf 2 mismatch");
    require(nodes[3].is_leaf() && nodes[3].right_child_index == 3, "Leaf 3 mismatch");
    require(nodes[4].is_leaf() && nodes[4].right_child_index == 4, "Leaf 4 mismatch");
    require(nodes[5].left_child_index == 0 && nodes[5].right_child_index == 1, "Internal node 5 mismatch");
    require(nodes[6].left_child_index == 2 && nodes[6].right_child_index == 3, "Internal node 6 mismatch");
    require(nodes[7].left_child_index == 6 && nodes[7].right_child_index == 4, "Internal node 7 mismatch");
    require(nodes[8].left_child_index == 5 && nodes[8].right_child_index == 7, "Root node mismatch");

    validate_binary_topology(nodes, scene);
  }

  template <unsigned int Arity>
  void require_child_indices_eq(const WideBVHNode<Arity>& node, const std::array<unsigned int, Arity>& expected, const std::string& label)
  {
    for (unsigned int slot = 0; slot < Arity; ++slot) {
      if (node.child_indices[slot] == expected[slot]) continue;

      std::ostringstream out;
      out << label << " child index mismatch at slot " << slot;
      throw std::runtime_error(out.str());
    }
  }

  void test_hploc_wide4_topology(cudaStream_t stream)
  {
    const TestScene scene = make_test_scene();
    const std::vector<BVHNode> binary_nodes = build_binary_bvh(stream, scene);
    const std::vector<WideBVHNode4> wide_nodes = build_wide_bvh<4>(stream, binary_nodes, scene.n_faces());

    require(wide_nodes.size() == 2, "Wide4 BVH must contain exactly two nodes for the test scene");
    require(wide_nodes[0].valid_mask == 0xFu, "Wide4 root valid mask mismatch");
    require(wide_nodes[0].primitive_mask == 0xEu, "Wide4 root primitive mask mismatch");
    require_child_indices_eq<4>(wide_nodes[0], std::array<unsigned int, 4>{1u, 2u, 4u, 3u}, "Wide4 root");
    require(wide_nodes[1].valid_mask == 0x3u, "Wide4 child valid mask mismatch");
    require(wide_nodes[1].primitive_mask == 0x3u, "Wide4 child primitive mask mismatch");
    require_child_indices_eq<4>(wide_nodes[1], std::array<unsigned int, 4>{0u, 1u, INVALID_INDEX, INVALID_INDEX}, "Wide4 child");

    validate_wide_topology(wide_nodes, scene);
  }

  void test_hploc_wide8_topology(cudaStream_t stream)
  {
    const TestScene scene = make_test_scene();
    const std::vector<BVHNode> binary_nodes = build_binary_bvh(stream, scene);
    const std::vector<WideBVHNode8> wide_nodes = build_wide_bvh<8>(stream, binary_nodes, scene.n_faces());

    require(wide_nodes.size() == 1, "Wide8 BVH must collapse into a single node for the test scene");
    require(wide_nodes[0].valid_mask == 0x1Fu, "Wide8 root valid mask mismatch");
    require(wide_nodes[0].primitive_mask == 0x1Fu, "Wide8 root primitive mask mismatch");
    require_child_indices_eq<8>(
        wide_nodes[0],
        std::array<unsigned int, 8>{0u, 2u, 4u, 3u, 1u, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX},
        "Wide8 root");

    validate_wide_topology(wide_nodes, scene);
  }

  void test_hploc_large_binary_topology(cudaStream_t stream)
  {
    const TestScene scene = make_large_test_scene();
    const std::vector<BVHNode> nodes = build_binary_bvh(stream, scene);

    require(scene.n_faces() == 33u, "Large binary test scene must contain 33 primitives");
    require(nodes.size() == 65u, "Large binary BVH must contain 65 nodes");
    validate_binary_topology(nodes, scene);
  }

  void test_hploc_large_wide4_topology(cudaStream_t stream)
  {
    const TestScene scene = make_large_test_scene();
    const std::vector<BVHNode> binary_nodes = build_binary_bvh(stream, scene);
    const std::vector<WideBVHNode4> wide_nodes = build_wide_bvh<4>(stream, binary_nodes, scene.n_faces());

    require(scene.n_faces() == 33u, "Large wide4 test scene must contain 33 primitives");
    require(wide_nodes.size() > 1u, "Large wide4 BVH should contain more than one node");
    require(wide_nodes.size() < binary_nodes.size(), "Wide4 BVH should be smaller than the binary BVH");
    validate_wide_topology(wide_nodes, scene);
  }

  void test_hploc_large_wide8_topology(cudaStream_t stream)
  {
    const TestScene scene = make_large_test_scene();
    const std::vector<BVHNode> binary_nodes = build_binary_bvh(stream, scene);
    const std::vector<WideBVHNode8> wide_nodes = build_wide_bvh<8>(stream, binary_nodes, scene.n_faces());

    require(scene.n_faces() == 33u, "Large wide8 test scene must contain 33 primitives");
    require(wide_nodes.size() > 1u, "Large wide8 BVH should contain more than one node");
    require(wide_nodes.size() < binary_nodes.size(), "Wide8 BVH should be smaller than the binary BVH");
    validate_wide_topology(wide_nodes, scene);
  }
}  // namespace

int main()
{
  cudaStream_t stream = nullptr;

  try {
    int device_count = 0;
    TEST_CUDA(cudaGetDeviceCount(&device_count));
    require(device_count > 0, "No CUDA devices available");

    TEST_CUDA(cudaSetDevice(0));
    TEST_CUDA(cudaStreamCreate(&stream));

    test_hploc_binary_topology(stream);
    test_hploc_wide4_topology(stream);
    test_hploc_wide8_topology(stream);
    test_hploc_large_binary_topology(stream);
    test_hploc_large_wide4_topology(stream);
    test_hploc_large_wide8_topology(stream);

    TEST_CUDA(cudaStreamDestroy(stream));
    std::cout << "All H-PLOC topology tests passed" << std::endl;
    return 0;
  } catch (const std::exception& ex) {
    if (stream != nullptr) cudaStreamDestroy(stream);
    std::cerr << "Topology test failed: " << ex.what() << std::endl;
    return 1;
  }
}
