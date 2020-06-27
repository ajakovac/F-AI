#include <iostream>
#include "OpenCLImplementation/GPUDevice.hpp"
#include "OpenCLImplementation/GPUVector.hpp"
#include "OpenCLImplementation/GPUCompute.hpp"
#include "OpenCLImplementation/DHVector.hpp"
#include "Timer.hpp"

#include "Rnd.hpp"

#include <ranges>
#include <sstream>
#include <fstream>

#include <numeric>

template<typename T>
std::ostream & operator<<(std::ostream &s, const std::vector<T> &x) {
    s << "(";
    if (x.size() > 1) for ( auto &v : x | std::views::take(x.size()-1)) s << v << ", ";
    if (x.size() > 0) s << x.back();
    s << ")";
    return s;
}

std::string read_file(const std::string& fname) {
	std::ifstream in(fname);
    std::ostringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}

int main(int, char **) try {

	Timer timer;
	auto dist = normal_dist(1.0, 1.0);
	auto dister = [=](double){return dist();};

	std::cerr.precision(10);
	std::cout.precision(10);

	GPUDevice gpu(0,0);

	GPUCompute comp(gpu, 0);
	comp.SetWorkGroupSize(256);

	std::size_t n = 60731253;

	DHVector<double> A({&gpu, 0}, n, 1.1);
	std::transform(A.begin(), A.end(), A.begin(), dister);
	DHVector<double> B(&gpu, n, 1.0);	
	DHVector<double> buffer(&gpu, n, 0.0);
	DHVector<double> buffer_small(&gpu, (n-1)/comp.WorkGroupSize() + 1, 0.0);

	timer.Record("CPU allocation.");

	//std::vector<double> Av(n, 1.1); GPUVector<double> A(Av, gpu, 0); A.ToGPU();
	//std::vector<double> Bv(n, 1.0); GPUVector<double> B(Bv, gpu, 0); B.ToGPU();
	//std::vector<double> bufv(n, 0.0); GPUVector<double> buffer(bufv, gpu, 0); buffer.ToGPU();
	//std::vector<double> bufsmallv( (n-1)/comp.WorkGroupSize() + 1, 0.0);
	//GPUVector<double> buffer_small(bufsmallv, gpu); buffer_small.ToGPU();
	
	std::cerr << "A to gpu.\n";
	A.ToGPU();
	std::cerr << "B to gpu.\n";
	B.ToGPU();
	std::cerr << "buffer to gpu.\n";
	buffer.ToGPU();
	std::cerr << "buffer_small to gpu.\n";
	buffer_small.ToGPU();

	timer.Record("GPU allocation.");

	comp.Multiply(A, B, buffer);
	comp.Finish();
	
	timer.Record("GPU calculation.");


	//A[0] = 123;
	//buffer[0] = 123;


	A.FromGPU();
	B.FromGPU();
	buffer.FromGPU();
	buffer_small.FromGPU();

	timer.Record("Copy back to CPU.");

    //std::cerr << "from gpu sikeres\n";
    std::cout << A[0] << "\n";
	std::cout << B[0] << "\n";	
	std::cout << buffer[0] << "\n";
	//std::cout << buffer_small << "\n";
	std::cout << buffer_small[0] << "\n";

	std::cerr << timer << "\n";

	Timer timer2;

	std::vector<double> a(n, 1.1);
	std::transform(a.begin(), a.end(), a.begin(), dister);
	std::vector<double> b(n, 1.0);
	std::vector<double> c(n, 1.0);

	timer2.Record("Allocate CPU");
	//std::cout << std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
	std::transform(a.begin(), a.end(), b.begin(), c.begin(), [](auto x, auto y){return x*y;});
	timer2.Record("CPU calculation.");

	std::cerr << timer2 << "\n";

	return 0;

/*
	auto kernel_code2 = read_file("OpenCLImplementation/Kernels/calculator.cl");
	gpu.Execute(kernel_code2, "ScalarMultiply", 0, {0,0,0}, {512}, {32}, 
		    A.GetBuffer(), B.GetBuffer(), buffer.GetBuffer(),
		    buffer_small.GetBuffer(), 512);

	buffer.FromGPU();
	buffer_small.FromGPU();
	std::cout<< bufv << "\n";
	std::cout<< bufsmallv << "\n";



	{
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[1];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
 
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
 
 
    cl::Context context({default_device});
 
    cl::Program::Sources sources;
 
    // kernel calculates for each element C=A+B
    std::string kernel_code=
            "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
            "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
            "   }                                                                               ";
    sources.push_back({kernel_code2.c_str(),kernel_code2.length()});
 
    cl::Program program(context,sources);
    if(program.build({default_device}, "-cl-std=CL2.0")!=CL_SUCCESS){
      if (CL_BUILD_ERROR == program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(default_device))
	std::cout << "CL_BUILD_ERROR\n";
      if (CL_BUILD_SUCCESS == program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(default_device))
	std::cout << "CL_BUILD_SUCCESS\n";
      if (CL_BUILD_IN_PROGRESS == program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(default_device))
	std::cout << "CL_BUILD_IN_PROGRESS\n";
      if (CL_BUILD_NONE == program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(default_device))
	std::cout << "CL_BUILD_NONE\n";

      
      std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    int N = 512;
 
 
    // create buffers on the device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(double)*N);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(double)*N);
    cl::Buffer buffer_buffer(context,CL_MEM_READ_WRITE,sizeof(double)*N);
    cl::Buffer buffer_small(context,CL_MEM_READ_WRITE,sizeof(double)*2);
 
    std::vector<double> A(N, 1.0);
    std::vector<double> B(N, 1.0);
    std::vector<double> sm(2, 1.0);
 
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);
 
    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(double)*N,A.data());
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(double)*N,B.data());
    queue.enqueueWriteBuffer(buffer_buffer,CL_TRUE,0,sizeof(double)*N,B.data());
    queue.enqueueWriteBuffer(buffer_small,CL_TRUE,0,sizeof(double)*2, sm.data());
 

    
    //alternative way to run the kernel
    cl::Kernel kernel_add=cl::Kernel(program,"ScalarMultiply");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_buffer);
    kernel_add.setArg(3,buffer_small);
    kernel_add.setArg(4,N);

    cl::Event k_events[2]; 

    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(N), cl::NDRange(32), NULL, &k_events[0]);
	k_events[0].wait();    
    queue.finish();


    
    std::vector<double> C(N, 1.0);
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_buffer,CL_TRUE,0,sizeof(double)*N, C.data());
    queue.enqueueReadBuffer(buffer_small,CL_TRUE,0,sizeof(double)*2, sm.data());
 
    std::cout << C << "\n";
    std::cout << sm << "\n";
 
    return 0;
	}
*/
} 
catch(Error &e) {
	std::cerr << e.error_message << std::endl;
}
