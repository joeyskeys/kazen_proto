#pragma once

class DictLike {
public:
    virtual void* address_of(const std::string& name) const = 0;
}