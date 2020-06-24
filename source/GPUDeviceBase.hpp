#ifndef F_AI_GPUDEVICEBASE_HPP_
#define F_AI_GPUDEVICEBASE_HPP_

#include <iostream>
#include <CL/cl.hpp>
#include <string>

enum class Implementation {
    OpenCL, CUDA
};

std::string ToString(Implementation imp) {
    switch(imp) {
        case Implementation::OpenCL: return "OpenCL";
        case Implementation::CUDA: return "CUDA";
        default: return "Unknown";        
    };
}

class GPUDeviceBase {
public:

    virtual std::string PlatformName() const = 0;
    virtual std::string DeviceName() const = 0;

    virtual Implementation Impl() const = 0;

};


#endif  // F_AI_GPUDEVICEBASE_HPP_