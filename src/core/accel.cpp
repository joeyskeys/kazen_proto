#include <limits>

#include "accel.h"
#include "material.h"
#include "sampling.h"

void Accelerator::add_hitable(std::shared_ptr<Hitable>&& h) {
    bound = bound_union(bound, h->bbox());
    hitables->emplace_back(h);
}

bool Accelerator::intersect(const Ray& r, Intersection& isect) const {
    bool hit = false;
    Intersection tmp_sect;
    tmp_sect.ray_t = std::numeric_limits<float>::max();
    float curr_t;
    auto tmax = r.tmax;
    auto tmin = r.tmin;

    for (auto& h : *hitables) {
        if (h->intersect(r, tmp_sect) && tmp_sect.ray_t < isect.ray_t) {
            hit = true;
            tmax = tmp_sect.ray_t;
            isect = tmp_sect;
        }
    }

    return hit;
}

bool Accelerator::intersect(const Ray& r, float& t) const {
    bool hit = false;
    float tmp_t = std::numeric_limits<float>::max();

    for (auto& h : *hitables) {
        if (h->intersect(r, tmp_t) && tmp_t < t) {
            hit = true;
            t = tmp_t;
        }
    }

    return hit;
}

void Accelerator::add_sphere(std::shared_ptr<Sphere>& s) {
    bound = bound_union(bound, s->bbox());
    hitables->emplace_back(s);
}

void Accelerator::add_quad(std::shared_ptr<Quad>& q) {
    bound = bound_union(bound, q->bbox());
    hitables->emplace_back(q);
}

void Accelerator::add_triangle(std::shared_ptr<Triangle>& t) {
    bound = bound_union(bound, t->bbox());
    hitables->emplace_back(t);
}

void Accelerator::add_trianglemesh(std::shared_ptr<TriangleMesh>& t) {
    bound = bound_union(bound, t->bbox());
    hitables->emplace_back(t);
}

void Accelerator::print_info() const {
    std::cout << fmt::format("List Accelerator with {} objects", size()) << std::endl;
    for (auto& objptr : *hitables) {
        std::cout << "\t";
        objptr->print_info();
        std::cout << std::endl;
    }
}

inline bool box_compare(const std::shared_ptr<Hitable>& a, const std::shared_ptr<Hitable>& b, int axis) {
    AABBf box_a;
    AABBf box_b;

    return a->bbox().min[axis] < b->bbox().min[axis];
}

class BVHNode : public Hitable {
public:
    BVHNode(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end, int l=0);

    bool intersect(const Ray& r, Intersection& isect) const override;
    bool intersect(const Ray& r, float& t) const override;
    void print_bound() const override;

    void print_info() const override;

private:
    std::shared_ptr<Hitable> children[2];
    int level;
};

//BVHAccel::BVHAccel(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end) {
BVHNode::BVHNode(std::vector<std::shared_ptr<Hitable>>& hitables, size_t start, size_t end, int l)
    : level(l)
{
    int axis = randomi(2);

    auto comparator = [&axis](auto a, auto b) {
        return box_compare(a, b, axis);
    };

    size_t object_span = end - start;
    assert(object_span > 0);

    if (object_span == 1) {
        children[0] = children[1] = hitables[start];
    }
    else if (object_span == 2) {
        if (comparator(hitables[start], hitables[start + 1])) {
            children[0] = hitables[start];
            children[1] = hitables[start + 1];
        }
        else {
            children[0] = hitables[start + 1];
            children[1] = hitables[start];
        }
    }
    else {
        std::sort(hitables.begin() + start, hitables.begin() + end, comparator);
        auto mid = start + object_span / 2;
        children[0] = std::make_shared<BVHNode>(hitables, start, mid, l + 1);
        children[1] = std::make_shared<BVHNode>(hitables, mid, end, l + 1);
    }

    bound = bound_union(children[0]->bbox(), children[1]->bbox());
}

bool BVHNode::intersect(const Ray& r, Intersection& isect) const {
    if (!bound.intersect(r)) return false;

    bool hit_0 = children[0]->intersect(r, isect);
    bool hit_1 = children[1]->intersect(r, isect);

    return hit_0 || hit_1;
}

bool BVHNode::intersect(const Ray& r, float& t) const {
    if (!bound.intersect(r))
        return false;
    
    bool hit_0 = children[0]->intersect(r, t);
    bool hit_1 = children[1]->intersect(r, t);

    return hit_0 || hit_1;
}

void BVHNode::print_bound() const {
    std::cout << "bvh node bound : " << bound;

    children[0]->print_bound();
    children[1]->print_bound();
}

void BVHNode::print_info() const {
    for (int i = 0; i < level; i++)
        std::cout << "\t";
    std::cout << "BVHNode:" << std::endl;

    children[0]->print_info();
    children[1]->print_info();
}

void BVHAccel::build() {
    root = std::make_shared<BVHNode>(*hitables, 0, hitables->size());
}

bool BVHAccel::intersect(const Ray& r, Intersection& isect) const {
    return root->intersect(r, isect);
}

bool BVHAccel::intersect(const Ray& r, float& t) const {
    return root->intersect(r, t);
}

void BVHAccel::print_info() const {
    root->print_info();
}

EmbreeAccel::EmbreeAccel(std::vector<std::shared_ptr<Hitable>>* hs)
    : Accelerator(hs)
{
    m_device = rtcNewDevice(nullptr);
    if (!m_device)
        std::cout << "Error : " << rtcGetDeviceError(nullptr)
            << "cannot create devic" << std::endl;
    
    m_scene = rtcNewScene(m_device);
    rtcSetSceneFlags(m_scene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(m_scene, RTC_BUILD_QUALITY_HIGH);
}

EmbreeAccel::~EmbreeAccel() {
    if (m_scene)
        rtcReleaseScene(m_scene);
    for (auto geom : m_geoms)
        rtcReleaseScene(geom);

    if (m_device)
        rtcReleaseDevice(m_device);
}

void EmbreeAccel::add_sphere(std::shared_ptr<Sphere>& s) {
    Accelerator::add_sphere(s);

    RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    // This is better but it causes a crash..
    //rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
        //RTC_FORMAT_FLOAT4, s->center_n_radius.data(), 0, sizeof(Vec4f), 1);
    Vec4f* cnr = reinterpret_cast<Vec4f*>(rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(Vec4f), 1));
    //cnr[0] = s->center_n_radius;
    s->get_world_position(cnr);
    rtcAttachGeometryByID(m_scene, geom, hitables->size() - 1);
    rtcReleaseGeometry(geom);
    rtcCommitGeometry(geom);
}

void EmbreeAccel::add_quad(std::shared_ptr<Quad>& q) {
    Accelerator::add_quad(q);

    RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_QUAD);
    auto vert_ptr = rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vec3f), 4);
    q->get_verts(vert_ptr);
    uint* indice_ptr = reinterpret_cast<uint*>(rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, 4 * sizeof(uint), 1));
    indice_ptr[0] = 0;
    indice_ptr[1] = 1;
    indice_ptr[2] = 2;
    indice_ptr[3] = 3;
    rtcCommitGeometry(geom);
    rtcAttachGeometryByID(m_scene, geom, hitables->size() - 1);
    rtcReleaseGeometry(geom);
}

void EmbreeAccel::add_triangle(std::shared_ptr<Triangle>& t) {
    Accelerator::add_triangle(t);

    RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
        RTC_FORMAT_FLOAT3, t->verts, 0, sizeof(Vec3f), 3);
    uint* indice_ptr = reinterpret_cast<uint*>(rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint), 1));
    indice_ptr[0] = 0;
    indice_ptr[1] = 1;
    indice_ptr[2] = 2;
    rtcCommitGeometry(geom);
    rtcAttachGeometryByID(m_scene, geom, hitables->size() - 1);
    rtcReleaseGeometry(geom);
}

void EmbreeAccel::add_trianglemesh(std::shared_ptr<TriangleMesh>& t) {
    Accelerator::add_trianglemesh(t);

    // Seems rtcSetGeometryTransform can only be applied to instance
    // So wrap the geometry into a sub-scene and make an instance of it
    // Then apply transform to the instance
    auto subscene = rtcNewScene(m_device);
    rtcSetSceneFlags(subscene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(subscene, RTC_BUILD_QUALITY_HIGH);
    //m_subscenes.push_back(subscene);

    RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
        RTC_FORMAT_FLOAT3, t->verts.data(), 0, sizeof(Vec3f), t->verts.size());
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
        RTC_FORMAT_UINT3, t->indice.data(), 0, sizeof(Vec3i), t->indice.size());
    
    /*
    // TODO : find out if there's any option to let embree calculate the interpolated
    // uv directly.
    // If triangle mesh have uv
    if (t->uvs.size() > 0) {
        rtcSetGeometryVertexAttributeCount(geom, 2);
        rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1,
            RTC_FORMAT_FLOAT2, t->uvs.data(), 0, sizeof(Vec2f), t->uvs.size());
        if (t->uv_idx.size() > 0) {
            rtcSetGeometryTopologyCount(geom, 2);
            rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1,
                RTC_FORMAT_UINT, t->uv_idx.data(), 0, sizeof(int), t->uv_idx.size());
            rtcSetGeometryVertexAttributeTopology(geom, 1, 1);
        }
    }
    */

    rtcCommitGeometry(geom);
    //rtcAttachGeometryByID(m_scene, geom, hitables->size() - 1);
    
    auto geom_id = rtcAttachGeometry(subscene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(subscene);

    // Ideally the instance geometry should be stored in the class and be applied
    // transform to later on, for animation between frames.
    // Or totally rebuild scene from ground up will be kinda a waste of time. So
    // leave it here as a
    // TODO : store instance and apply transform between frames.
    RTCGeometry instance = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(instance, subscene);
    rtcSetGeometryTimeStepCount(instance, 1);
    rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, t->local_to_world.mat.data());
    rtcCommitGeometry(instance);
    //auto ins_id = rtcAttachGeometry(m_scene, instance);
    //rtcReleaseGeometry(instance);
    rtcReleaseScene(subscene);

    m_geoms.emplace(t->name, instance);
}

void EmbreeAccel::add_instances(const std::string& name, std::vector<std::string>& instance_names, const Transform& trans) {
    auto subscene = rtcNewScene(m_device);
    rtcSetSceneFlags(subscene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(subscene, RTC_BUILD_QUALITY_HIGH);
    for (const auto& instance_name : instance_names) {
        auto geom = m_geoms.find(instance_name);
        assert(geom != m_geoms.end());
        rtcAttachGeometry(subscene, *geom);
    }
    rtcCommitScene(subscene);
    RTCGeometry instance = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(instance, subscene);
    rtcSetGeometryTimeStepCount(instance, 1);
    rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, trans->local_to_world.mat.data());
    rtcCommitGeometry(instance);
    rtcReleaseScene(subscene);

    m_geoms.emplace(name, instance);
}

void EmbreeAccel::build(const std::vector<std::string>& instance_names) {
    for (const auto& instance_name : instance_names) {
        auto geom = m_geoms.find(instance_name);
        assert(geom != m_geoms.end());
        rtcAttachGeometry(m_scene, *geom);
    }
    rtcCommitScene(m_scene);
}

bool EmbreeAccel::intersect(const Ray& r, Intersection& isect) const {
    RTCIntersectArguments iargs;
    rtcInitIntersectArguments(&iargs);

    RTCRayHit rayhit;
    rayhit.ray.org_x = r.origin.x();
    rayhit.ray.org_y = r.origin.y();
    rayhit.ray.org_z = r.origin.z();
    rayhit.ray.dir_x = r.direction.x();
    rayhit.ray.dir_y = r.direction.y();
    rayhit.ray.dir_z = r.direction.z();
    rayhit.ray.tnear = r.tmin;
    rayhit.ray.tfar = r.tmax;
    rayhit.ray.mask = 1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(m_scene, &rayhit, &iargs);
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        isect.ray = const_cast<Ray*>(&r);
        isect.ray_t = rayhit.ray.tfar;
        isect.wi = -r.direction;
        isect.P = r.at(isect.ray_t);
        isect.N = normalize(Vec3f(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));
        isect.uv = Vec2f(rayhit.hit.u, rayhit.hit.v);
        isect.geom_id = rayhit.hit.geomID;
        isect.prim_id = rayhit.hit.primID;
        //isect.shape = reinterpret_cast<Shape*>(hitables->at(rayhit.hit.geomID).get());
        /*
         * RTCHit object have the instID array field which will hold the instance list
         * Suppose the scene is:
         * Scene(id = 0)
         * |-> Instance(geomID = 0)
         * |    |-> Sub Scene(id = 1)
         * |    |   |->Mesh1(geomID = 0)
         * |    |   |->Mesh2(geomID = 1)
         * |-> Instance(geomID = 1)
         * |    |   |-> Instance(geomID = 0)
         * |    |   |   |-> Sub Scene(id = 2)
         * |    |   |   |   |->Mesh3(geomID = 0)
         * 
         * If we hit Mesh3, the instID array of RTCHit object will hold 1, 0,
         * terminated with RTC_INVALID_GEOMETRY_ID.
         * And geomID will hold the value 0
         * 
         */
        // TODO : This is a convinient solution for adding transform and the
        // scene description does not support instancing for now.
        // But nested instancing need to handle instID array correctly.
        isect.shape = reinterpret_cast<Shape*>(m_subscenes.at(rayhit.hit.instID[0]).second.get());
        isect.shape->post_hit(isect);
        isect.frame = Frame(isect.shading_normal);
    
        return true;
    }

    return false;
}

bool EmbreeAccel::intersect(const Ray& r, float& t) const {
    RTCIntersectArguments iargs;
    rtcInitIntersectArguments(&iargs);

    RTCRayHit rayhit;
    rayhit.ray.org_x = r.origin.x();
    rayhit.ray.org_y = r.origin.y();
    rayhit.ray.org_z = r.origin.z();
    rayhit.ray.dir_x = r.direction.x();
    rayhit.ray.dir_y = r.direction.y();
    rayhit.ray.dir_z = r.direction.z();
    rayhit.ray.tnear = r.tmin;
    rayhit.ray.tfar = r.tmax;
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(m_scene, &rayhit, &iargs);
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        t = rayhit.ray.tfar;
        return true;
    }
    return false;
}

void EmbreeAccel::print_info() const {
    std::cout << "Embree Accelerator" << std::endl;
}

OptixAccel::OptixAccel(const OptixDeviceContext& c)
    : ctx(c)
{}

OptixAccel::~OptixAccel() {
    if (gas_handles.size() > 0) {
        for (const auto& handle_pair : handles)
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(handle_pair.second)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(root_buf)));
    }
}

void OptixAccel::add_sphere(std::shared_ptr<Sphere& s) {
    // GAS
    OptixAccelBuildOptions accel_options {
        .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
        .operation = OPTIX_BUILD_OPERATION_BUILD
    };

    CUdeviceptr d_vertex_buf;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_buf), sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertex_buf), &s.center_n_radius,
        sizeof(float3), cudaMemcpyHostToDevice));

    CUdeviceptr d_radius_buf;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius_buf), sizeof(float)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius_buf), &s.center_n_radius[3],
        sizeof(float), cudaMemcpyHostToDevice));

    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    input.sphereArray.vertexBuffers = &d_vertex_buf;
    input.sphereArray.numVertices = 1;
    input.sphereArray.radiusBuffers = &d_radius_buf;

    uint32_t input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    input.sphereArray.flags = input_flags;
    input.sphereArray.numSbtRecords = 1;

    OptixAccelBufferSizes buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &accel_options, &sphere_input, 1, &buf_sizes));
    CUdeviceptr d_temp_buf;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buf), buf_sizes.tempSizeInBytes));

    CUdeviceptr d_non_compact_output;
    size_t compact_size_offset = round_up<size_t>(buf_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_non_compact_output),
        compact_size_offset + 8));

    OptixAccelEmitDesc emit_prop = {};
    emit_prop.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_prop.result = (CUdeviceptr)((char*)d_non_compact_output + compact_size_offset);

    OptixTraversableHandle handle;
    OPTIX_CHECK(optixAccelBuild(ctx,
                                0,
                                &accel_options, &input,
                                1,
                                d_temp_buf, buf_sizes.tempSizeInBytes,
                                d_non_compact_output, buf_sizes.outputSizeInBytes, &handle,
                                &emit_prop,
                                1));

    CUdeviceptr d_output;
    CUDA_CHECK(cudaFree((void*)d_temp_buf));
    CUDA_CHECK(cudaFree((void*)d_vertex_buf));
    CUDA_CHECK(cudaFree((void*)d_radius_buf));

    size_t compact_size;
    CUDA_CHECK(cudaMemcpy(&compact_size, (void*)emit_prop.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compact_size < buf_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), compact_size));
        OPTIX_CHECK(optixAccelCompact(ctx, 0, handle, d_output, compact_size, &handle));
        CUDA_CHECK(cudaFree((void*)d_non_compact_output));
    }
    else {
        d_output = d_non_compact_output;
    }

    gas_handles.push_back(handle);
    gas_output_bufs.push_back(d_output);

    // IAS information
    OptixInstance inst;
    const auto optix_transform = base::transpose(s->local_to_world.mat);
    memcpy(inst.transform, optix_transform.data(), sizeof(float) * 12);
    inst.instanceId             = s->id;
    inst.visibilityMask         = 255;
    inst.sbtOffset              = 0;
    inst.flags                  = OPTIX_INSTANCE_FLAG_NONE;
    inst.OptixTraversableHandle = handle;
    instances.push_back(inst);
}

void OptixAccel::add_quad(std::shared_ptr<Quad>& q) {

}

void OptixAccel::add_triangle(std::shared_ptr<Triangle>& t) {

}

void OptixAccel::add_trianglemesh(std::shared_ptr<TriangleMesh>& t) {
    CUdeviceptr d_vertices, d_indices, d_trans;
    const size_t vertices_size = t->verts.size() * sizeof(Vec3f);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices), t->verts.data(),
        vertices_size, cudaMemcpyHostToDevice));
    const size_t indices_size = t->indice.size() * sizeof(Vec3i);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices), t->indice.data(),
        indices_size, cudaMemcpyHostToDevice));
    const size_t trans_size = sizeof(float) * 12;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_trans), trans_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_trans), t->local_to_world.mat.transpose().data(),
        trans_size, cudaMemcpyHostToDevice));

    uint32_t input_flags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS};
    OptixBuildInputTriangleArray gas_input {
        .vertexBuffers = &d_vertices,
        .numVertices = static_cast<uint32_t>(t->verts.size()),
        .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
        .vertexStrideInBytes = sizeof(float3),
        .indexBuffer = &d_indices,
        .numIndexTriplets = static_cast<uint32_t>(t->indice.size()),
        .indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
        .indexStrideBytes = sizeof(uint3),
        .preTransform = d_trans,
        .flags = input_flags,
        .numSbtRecords = 1,
        .sbtIndexOffsetBuffer = nullptr,
        .sbtIndexOffsetSizeInBytes = sizeof(uint32_t),
        .sbtIndexOffsetStrideInBytes = sizeof(uint32_t)
    };

    OptixAccelBuildOptions gas_accel_options = {
        .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
        .operation = OPTIX_BUILD_OPERATION_BUILD
    };

    OptixAccelBufferSizes gas_buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        ctx,
        &gas_accel_options,
        &gas_input,
        1,
        &gas_buf_sizes
    ));

    CUdeviceptr d_tmp_buf;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp_buf), gas_buf_sizes.tempSizeInBytes));

    CUdeviceptr d_non_compact_output;
    size_t compact_size_offset = round_up<size_t>(gas_buf_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_non_compact_output), compact_size_offset + 8));

    OptixAccelEmitDesc emit_prop = {
        .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
        .result = (CUdeviceptr)((char*)d_non_compact_output + compact_size_offset);
    };

    OptixTraversableHandle gas_handle;
    OPTIX_CHECK(optixAccelBuild(ctx,
                                0,
                                &gas_accel_options,
                                &gas_input,
                                1,
                                d_tmp_buf,
                                gas_buf_sizes.tempSizeInBytes,
                                d_non_compact_output,
                                gas_buf_sizes.outputSizeInBytes,
                                &gashandle,
                                &emit_prop,
                                1));
    
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buf)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));

    size_t compact_size;
    CUDA_CHECK(cudaMemcpy(&compact_size, (void*)emit_prop.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    CUdeviceptr d_output;
    if (compact_size < buf_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), compact_size));
        OPTIX_CHECK(optixAccelCompact(ctx, 0, gas_handle, d_output, compact_size, &gas_handle));
        CUDA_CHECK(cudaFree((void*)d_non_compact_output));
    }
    else {
        d_output = d_non_compact_output;
    }

    gas_handles.push_back(gas_handle);
    gas_output_bufs.push_back(d_output);
}

void OptixAccel::add_spheres(std::vector<std::shared_ptr<Sphere>& ss) {

}

OptixTraversableHandle OptixAccel::add_instances(const std::string& name, const std::vector<std::string>& handle_names, const Transform& trans) {
    // Create an array of instances input first
    CUdeviceptr d_inst;
    size_t inst_size = handle_names.size() * sizeof(OptixInstance);
    std::vector<OptixInstance> insts(handle_names.size());
    for (const auto& handle_name : handle_names) {
        auto handle = handles.find(handle_name);
        assert(handle != handles.end());

        OptixInstance inst;
        const auto optix_transform = base::transpose(trans.local_to_world.mat);
        memcpy(inst.transform, optix_transform.data(), sizeof(float) * 12);
        inst.instanceId             = handles.size();
        inst.visibilityMask         = 255;
        inst.sbtOffset              = 0;
        inst.flags                  = OPTIX_INSTANCE_FLAG_NONE;
        inst.OptixTraversableHandle = handle;

        insts.push_back(inst);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inst), inst_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_inst), insts.data(), inst_size));

    OptixBuildInput input {
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = d_inst,
            .numInstances = insts.size();
        }
    };

    OptixAccelBuildOptions accel_options {
        .buildFlags = OPTIX_BUILD_FLAG_NONE,
        .operation = OPTIX_BUILD_OPERATION_BUILD
    };

    OptixAccelBufferSizes buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &accel_options, &input, 1, &buf_sizes));
    CUdeviceptr d_tmp_buf, d_output;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp_buf), buf_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), buf_sizes.outputSizeInBytes));

    OptixTraversableHandle ias_handle;
    OPTIX_CHECK(optixAccelBuild(
        ctx,
        0,
        &accel_options,
        &input,
        1,
        d_tmp_buf,
        buf_sizes.tempSizeInBytes,
        d_output,
        buf_sizes.outputSizeInBytes,
        &ias_handle,
        nullptr,
        0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buf)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_inst)));

    handles.emplace_back(name, ias_handle, d_output);
}

void OptixAccel::build(const std::vector<std::string>& handle_names) {
    CUdeviceptr d_inst;
    size_t inst_size = instances.size() * sizeof(OptixInstance);
    std::vector<OptixInstance> insts(handle_names.size());
    for (const auto& handle_name : handle_names) {
        auto handle = handles.find(handle_name);
        assert(handle != handles.end());

        OptixInstance inst;
        const auto optix_transform = base::transpose(trans.local_to_world.mat);
        memcpy(inst.transform, optix_transform.data(), sizeof(float) * 12);
        inst.instanceId             = handles.size();
        inst.visibilityMask         = 255;
        inst.sbtOffset              = 0;
        inst.flags                  = OPTIX_INSTANCE_FLAG_NONE;
        inst.OptixTraversableHandle = handle;

        insts.push_back(inst);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inst), inst_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_inst), insts.data(), inst_size));

    OptixBuildInput input {
        .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .instanceArray = {
            .instances = d_inst,
            .numInstances = insts.size()
        }
    };
    OptixAccelBuildOptions accel_options {
        .buildFlags = OPTIX_BUILD_FLAG_NONE,
        .operation = OPTIX_BUILD_OPERATION_BUILD
    };

    OptixAccelBufferSizes buf_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &accel_options, &input, 1, &buf_sizes));
    CUdeviceptr d_tmp_buf, d_output;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp_buf), buf_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), buf_sizes.outputSizeInBytes));
    
    OPTIX_CHECK(optixAccelBuild(
        ctx,
        0,
        &accel_options,
        &input,
        1,
        d_tmp_buf,
        buf_sizes.tempSizeInBytes,
        d_output,
        buf_sizes.outputSizeInBytes,
        &root_handle,
        nullptr,
        0));
    
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tmp_buf)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_inst)));
}

void OptixAccel::print_info() {

}