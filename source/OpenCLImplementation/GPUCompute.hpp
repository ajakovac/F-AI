#ifndef F_AI_GPUCOMPUTE_HPP_
#define F_AI_GPUCOMPUTE_HPP_


#include "GPUDevice.hpp"
#include "GPUVector.hpp"
#include "DHVector.hpp"

#include <sstream>
#include <fstream>
#include <string>


namespace _detail {

}

class GPUCompute {
	GPUDevice* device;
	cl::Program* program = nullptr;
	std::size_t queue_n;
	std::size_t work_group_size = 256;
public:

	inline GPUCompute(GPUDevice& dev_in, std::size_t q_in) : device(&dev_in), queue_n(q_in) {
		program = device->Compile(read_file("OpenCLImplementation/Kernels/calculatorv2.cl"));
	}

	inline void SetWorkGroupSize(std::size_t n) {
		if (n > device->MaxWorkGroupSize()) {
			std::cerr << "Warning: GPUCompute::SetWorkGroupSize: given work group size exceeds maximum.\n";
			n = device->MaxWorkGroupSize();
		}
		if (n % 64 != 0) n = (n/64) * 64;
		if (n < 64) n = 64;
		work_group_size = n;
	}

	inline std::size_t WorkGroupSize() const {return work_group_size;}

	/*! \brief copy A to B. A, B and out must have a common size, THAT
		IS A MULTIPLE OF work_group_size (preferably a power of 2). */
	inline void Copy(GPUVector<double>& A, GPUVector<double>& B) {
		Copy(A, B, A.Size());
	}

	/*! \brief copy \a n elements of \a A to \a B. A, B and out must have a common size, THAT
		IS A MULTIPLE OF work_group_size (preferably a power of 2). */
	inline void Copy(GPUVector<double>& A, GPUVector<double>& B, std::size_t n) {
		cl::Kernel ker(*program, "copyTo");
		ker.setArg(0, A.GetBuffer());
		ker.setArg(1, B.GetBuffer());
		ker.setArg(2, (int)(n));
		std::cout << "copying with: " << (int)(n) << "\n";
        device->Queue(queue_n).enqueueNDRangeKernel(ker, cl::NullRange, cl::NDRange(adjustGlobalSize(n)), 
        											cl::NDRange(work_group_size), 0, 0);
	}

	/*! \brief Pointwise multiplication of \a A and \a B to \a out. B and out must be a least as large as A. */
	inline void Multiply(GPUVector<double>& A, GPUVector<double>& B, GPUVector<double>& out) {
		Multiply(A, B, out, A.Size());
	}

	/*! \brief Pointwise multiplication of \a A and \a B to \a out. A, B and out each must have a size of at least n. */
	inline void Multiply(GPUVector<double>& A, GPUVector<double>& B, GPUVector<double>& out, std::size_t n) {
		cl::Kernel ker(*program, "pointwiseMultiply");
		ker.setArg(0, A.GetBuffer());
		ker.setArg(1, B.GetBuffer());
		ker.setArg(2, out.GetBuffer());
		ker.setArg(3, (int)(n));
		ker.setArg(4, (int)(50));
		std::cout << "multiplying with: " << (int)(n) << "\n";
        device->Queue(queue_n).enqueueNDRangeKernel(ker, cl::NullRange, cl::NDRange(adjustGlobalSize(n)), 
        											cl::NDRange(work_group_size), 0, 0);
	}

	/*! \brief Sums the elements of \a A to buffer_small[0] and buffer[0]. \a A and \a buffer must have a common size. 
		\a buffer_small MUST have a size AT LEAST of ceil(A.size()/work_group_size). */
	inline void Sum(GPUVector<double>& A, GPUVector<double>& buffer, GPUVector<double>& buffer_small) {
		Copy(A, buffer);
        InlineSum(buffer, buffer_small);
	}

	inline void InlineSum(GPUVector<double>& buffer, GPUVector<double>& buffer_small) {
		std::size_t n = buffer.Size();
		while (n > 1) { 
			cl::Kernel ker(*program, "sumTo");
			ker.setArg(0, buffer.GetBuffer());
			ker.setArg(1, buffer_small.GetBuffer());
			ker.setArg(2, int(n));
			ker.setArg(3, sizeof(double) * work_group_size, NULL);
			//std::cerr << "adjusted to: " << adjustGlobalSize(n) << "\n";
        	device->Queue(queue_n).enqueueNDRangeKernel(ker, cl::NullRange, cl::NDRange(adjustGlobalSize(n)), 
        												cl::NDRange(work_group_size), 0, 0);
        	n = 1 + (n-1)/work_group_size;
        	Copy(buffer_small, buffer, n);
        	//std::cerr << "n = " << n << "\n";
    	}
	}

	inline void ScalarMultiply(GPUVector<double>& A, GPUVector<double>& B, 
							   GPUVector<double>& buffer, GPUVector<double>& buffer_small) {
		Multiply(A, B, buffer);
		InlineSum(buffer, buffer_small);
	}

	inline void Finish() {device->Queue(queue_n).finish();}

	// ---------------------------------------------------------------------------------------
    // DHVector wrapper:

	/*! \brief copy A to B. A, B and out must have a common size, THAT
	IS A MULTIPLE OF work_group_size (preferably a power of 2). */
	inline void Copy(DHVector<double>& A, DHVector<double>& B) {
		Copy(A.GPUVec(), B.GPUVec());
	}

	/*! \brief copy \a n elements of \a A to \a B. A, B and out must have a common size, THAT
		IS A MULTIPLE OF work_group_size (preferably a power of 2). */
	inline void Copy(DHVector<double>& A, DHVector<double>& B, std::size_t n) {
		Copy(A.GPUVec(), B.GPUVec(), n);
	}

	/*! \brief Pointwise multiplication of \a A and \a B to \a out. B and out must be a least as large as A. */
	inline void Multiply(DHVector<double>& A, DHVector<double>& B, DHVector<double>& out) {
		Multiply(A.GPUVec(), B.GPUVec(), out.GPUVec());
	}

	/*! \brief Pointwise multiplication of \a A and \a B to \a out. A, B and out each must have a size of at least n. */
	inline void Multiply(DHVector<double>& A, DHVector<double>& B, DHVector<double>& out, std::size_t n) {
		Multiply(A.GPUVec(), B.GPUVec(), out.GPUVec(), n);
	}

	/*! \brief Sums the elements of \a A to buffer_small[0] and buffer[0]. \a A and \a buffer must have a common size. 
		\a buffer_small MUST have a size AT LEAST of ceil(A.size()/work_group_size). */
	inline void Sum(DHVector<double>& A, DHVector<double>& buffer, DHVector<double>& buffer_small) {
		Sum(A.GPUVec(), buffer.GPUVec(), buffer_small.GPUVec());
	}

	inline void InlineSum(DHVector<double>& buffer, DHVector<double>& buffer_small) {
		InlineSum(buffer.GPUVec(), buffer_small.GPUVec());
	}

	inline void ScalarMultiply(DHVector<double>& A, DHVector<double>& B, 
							   DHVector<double>& buffer, DHVector<double>& buffer_small) {
		ScalarMultiply(A.GPUVec(), B.GPUVec(), buffer.GPUVec(), buffer_small.GPUVec());
	}

private:
	static std::string read_file(const std::string& fname) {
		std::ifstream in(fname);
    	std::ostringstream sstr;
    	sstr << in.rdbuf();
    	return sstr.str();
	}

	/*! \brief Creates a "global size", that is divisible by the work group size, and is 
		larger than the actual work group size. */
	std::size_t adjustGlobalSize(std::size_t actual_global_size) const {
		if (actual_global_size == 0) return 0;
		return (1 + (actual_global_size-1)/work_group_size)*work_group_size;
	}
};






#endif // F_AI_GPUCOMPUTE_HPP_