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

    RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
        RTC_FORMAT_FLOAT3, t->verts.data(), 0, sizeof(Vec3f), t->verts.size());
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
        RTC_FORMAT_UINT3, t->indice.data(), 0, sizeof(Vec3i), t->indice.size());
    rtcCommitGeometry(geom);
    rtcAttachGeometryByID(m_scene, geom, hitables->size() - 1);
    rtcReleaseGeometry(geom);
}

void EmbreeAccel::build() {
    rtcCommitScene(m_scene);
}

bool EmbreeAccel::intersect(const Ray& r, Intersection& isect) const {
    RTCIntersectContext isect_ctx;
    rtcInitIntersectContext(&isect_ctx);

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

    rtcIntersect1(m_scene, &isect_ctx, &rayhit);
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        isect.ray_t = rayhit.ray.tfar;
        isect.P = r.at(isect.ray_t);
        isect.N = Vec3f(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z).normalized();
        isect.uv = Vec2f(rayhit.hit.u, rayhit.hit.v);
        isect.geom_id = rayhit.hit.geomID;
        isect.prim_id = rayhit.hit.primID;
        isect.shape = reinterpret_cast<Shape*>(hitables->at(rayhit.hit.geomID).get());
        isect.shape->post_hit(isect);
        isect.frame = Frame(isect.shading_normal);
    
        return true;
    }

    return false;
}

bool EmbreeAccel::intersect(const Ray& r, float& t) const {
    RTCIntersectContext isect_ctx;
    rtcInitIntersectContext(&isect_ctx);

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

    rtcIntersect1(m_scene, &isect_ctx, &rayhit);
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        t = rayhit.ray.tfar;
        return true;
    }
    return false;
}

void EmbreeAccel::print_info() const {
    std::cout << "Embree Accelerator" << std::endl;
}