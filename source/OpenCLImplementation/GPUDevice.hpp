#ifndef F_AI_GPUDEVICE_HPP_
#define F_AI_GPUDEVICE_HPP_

#include <iostream>
#include <string>
#include <CL/cl.hpp>
#include "../Error.hpp"
#include "../GPUDeviceBase.hpp"

class GPUDevice : public GPUDeviceBase {
    std::vector<cl::Platform> all_platforms;
    std::vector<cl::Device> all_devices;
        
    cl::Platform platform;
    std::size_t platform_index;

    cl::Device device;
    std::size_t device_index;

    cl::Context context;
    //cl::Program::Sources sources;

    std::vector<cl::CommandQueue> queues;

 public:
 	inline GPUDevice() : GPUDevice(0, 0) {}

    inline GPUDevice(std::size_t nP, std::size_t nD) : platform_index(nP), device_index(nD),
        queues(0) {
        cl::Platform::get(&all_platforms);
        if(all_platforms.size()==0)
            throw(Error("GPUDevice: no platforms found. Check OpenCL installation!"));
        if (all_platforms.size() <= nP)
            throw(Error("GPUDevice: not enough platforms. Expected: " + std::to_string(nP) 
                + ", max: " + std::to_string(all_platforms.size() -1) + "."  ));
        platform=all_platforms[nP];  
        platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if(all_devices.size()==0)
            throw(Error(" No devices found. Check OpenCL installation!"));
        if (all_platforms.size() <= nD)
            throw(Error("GPUDevice: not enough devices. Expected: " + std::to_string(nD) 
                + ", max: " + std::to_string(all_devices.size()-1) + "."  ));
        device=all_devices[nD];

        context = cl::Context({device});
        queues.push_back(cl::CommandQueue(context, device));

        std::cout << "Using platform: "<<platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
	std::cout << "Using device: "<<device.getInfo<CL_DEVICE_NAME>()<<"\n";

    }
    // ---------------------------------------------------------------------------------------
    // Getters:

    inline std::string PlatformName() const override {return platform.getInfo<CL_PLATFORM_NAME>();}
    inline std::string Name() const override {return device.getInfo<CL_DEVICE_NAME>();}
    inline std::size_t MaxWorkGroupSize() const {return device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();}
    inline Implementation Impl() const override {return Implementation::OpenCL;}

    inline cl::Context& Context() {return context;}
    inline cl::Device& Device() {return device;}
    inline cl::CommandQueue& Queue(std::size_t n) {return queues[n];}


    // ---------------------------------------------------------------------------------------
    // 

    inline cl::Program* Compile(std::string const& code, std::string const& flags = "") { // -cl-std=CL2.0
        auto program = new cl::Program(context, { {code.c_str(), code.size()} } );
        if (program->build({device}, flags.c_str()) != CL_SUCCESS)
            throw Error(std::string("GPUDevice::Execute: Error building kernel code: ")
                        + program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
        return program;
    }


    template<typename... Args>
    inline void Execute(std::string const& kernel_code, 
                        std::string const& function,
                        std::size_t qid, 
                        cl::NDRange const& offset,
                        cl::NDRange const& global,
                        cl::NDRange const& local,
                        Args&&... args) {
        cl::Program *pr = Compile(kernel_code, "-cl-std=CL2.0");
        cl::Kernel ker(*pr, function.c_str());
        execute_help(ker, 0, std::forward<Args>(args)...);
        auto queue = queues[qid];
        queue.enqueueNDRangeKernel(ker, offset, global, local);
        queue.finish();
        delete pr;
    }


private:
    template<class Arg, typename... Args>
    inline void execute_help(cl::Kernel& ker, std::size_t n, Arg&& arg, Args&&... args) {
        ker.setArg(n, arg);
        execute_help(ker, ++n, std::forward<Args>(args)...);
    }

    inline void execute_help(cl::Kernel&, std::size_t) {}
};


#endif  // F_AI_GPUDEVICE_HPP_
