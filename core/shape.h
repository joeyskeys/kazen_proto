#pragma once

#include "transform.h"
#include "intersection.h"

class Shape {
public:
    Shape(const Transform& l2w);

    virtual Intersection intersect(const Ray& r) const = 0;
    
    // Members
    Transform world_to_local, local_to_world;
};

class Sphere : public Shape {
public:

};

class Triangle : public Shape {
public:

};

class TriangleMesh : public Shape {
public:

};