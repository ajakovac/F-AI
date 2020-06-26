#include <iostream>
#include "OpenCLImplementation/GPUDevice.hpp"
#include "OpenCLImplementation/GPUVector.hpp"

#include <ranges>
#include <sstream>
#include <fstream>

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

int main(int argc, char **argv) try {

	GPUDevice gpu;
	/*cl::CommandQueue& queue = gpu.Queue(0);

    std::vector<int> A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> B = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
    GPUVector<int> GPUA(A, gpu);
    GPUVector<int> GPUB(B, gpu);
    GPUA.ToGPU();
    GPUB.ToGPU();

	std::vector<int> C(10,0);
	GPUVector<int> GPUC(C, gpu);
	GPUC.ToGPU();

    // kernel calculates for each element C=A+B
    std::string kernel_code=
        "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
        "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                       "
        "   }                                                                                      ";

	/*
   	/*cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    cl::Program program(gpu.Context(),sources);
    if(program.build({gpu.Device()})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpu.Device())<<"\n";
        exit(1);
    }
    cl::Kernel simple_add(program, "simple_add");
    simple_add.setArg(0, GPUA.GetBuffer());
    simple_add.setArg(1, GPUB.GetBuffer());
    simple_add.setArg(2, GPUC.GetBuffer());
    queue.enqueueNDRangeKernel(simple_add,cl::NullRange,cl::NDRange(10),cl::NDRange(10));
    queue.finish();*/
	/*
    gpu.Execute(kernel_code, "simple_add", 0, cl::NullRange,cl::NDRange(10),cl::NDRange(10), 
		GPUA.GetBuffer(), GPUB.GetBuffer(), GPUC.GetBuffer());

    GPUC.FromGPU();

    std::cout << C << "\n";
    std::cout << "\n";*/
	//GPUDevice gpu(0,0);


	/*std::string kernel_code1=
    	"   void kernel simple_add(global const int* A, global const int* B, global int* C, int n){ \n"
        "       int u = get_global_id(0);                                                           \n"
        "       float t = A[u]+B[n - u];                                                            \n"
        "	    C[u]= 1000*sin(t);                                                                       \n"
        "   }                                                                                  ";

	std::string s = gpu.DeviceName();
	std::cout << s << "\n";

	std::vector<int> vec = {0,1,2,3,4,5,6,7,8,9};

	GPUVector<int> gv(vec, gpu);

	gv.ToGPU();

	gpu.Execute(kernel_code1, "simple_add", 0, {0,0,0}, {10}, {2},
		    gv.GetBuffer(), gv.GetBuffer(), gv.GetBuffer(), 9);

	gv.FromGPU();

	std::cout << vec << "\n\n\n";
	*/
	std::vector<double> Av(512, 1); GPUVector<double> A(Av, gpu); A.ToGPU();
	std::vector<double> Bv(512, 1); GPUVector<double> B(Bv, gpu); B.ToGPU();
	std::vector<double> bufv(512, 0.0); GPUVector<double> buffer(bufv, gpu); buffer.ToGPU();
	std::vector<double> bufsmallv(2, 0.0); 
	GPUVector<double> buffer_small(bufsmallv, gpu); buffer_small.ToGPU();

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
    cl::Platform default_platform=all_platforms[0];
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
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(N), cl::NDRange(32));
    queue.finish();


    
    std::vector<double> C(N, 1.0);
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_buffer,CL_TRUE,0,sizeof(double)*N, C.data());
    queue.enqueueReadBuffer(buffer_small,CL_TRUE,0,sizeof(double)*2, sm.data());
 
    std::cout << C << "\n";
    std::cout << sm << "\n";
 
    return 0;
	}

} 
catch(Error &e) {
	std::cerr << e.error_message << std::endl;
}
