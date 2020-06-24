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


	std::string kernel_code=
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

	gpu.Execute(kernel_code, "simple_add", 0, {0,0,0}, {10}, {2}, gv, gv, gv, 9);

	gv.FromGPU();

	std::cout << vec << "\n";

	/*std::vector<double> Av(512, 1.0); GPUVector<double> A(Av, gpu); A.ToGPU();
	std::vector<double> Bv(512, 1.0); GPUVector<double> B(Bv, gpu); B.ToGPU();
	std::vector<double> bufv(512, 0.0); GPUVector<double> buffer(bufv, gpu); buffer.ToGPU();
	std::vector<double> bufsmallv(2, 0.0); 
	GPUVector<double> buffer_small(bufsmallv, gpu); buffer_small.ToGPU();

	auto kernel_code2 = read_file("OpenCLImplementation/Kernels/calculator.cl");
	gpu.Execute(kernel_code2, "ScalarMultiply", 0, {0,0,0}, {512}, {32}, 
				A, B, buffer, buffer_small, 512);

	buffer.FromGPU();
	std::cout<< bufv[0] << "\n";*/

} 
catch(Error &e) {
	std::cerr << e.error_message << std::endl;
}