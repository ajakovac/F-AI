#ifndef F_AI_GPUVECTOR_HPP_
#define F_AI_GPUVECTOR_HPP_

#include "../GPUVectorBase.hpp"
#include "GPUDevice.hpp"

template<class T>
class GPUVector: GPUVectorBase<T> {
	cl::Buffer *buffer;
	std::size_t qid;
	using GPUVectorBase<T>::device;
	using GPUVectorBase<T>::cpuv;
	using GPUVectorBase<T>::size;
public: 

	inline GPUVector(std::vector<T>& cpu_vec, GPUDeviceBase& d, std::size_t qin = 0)
	 : GPUVectorBase<T>(cpu_vec, d), buffer(nullptr), qid(qin) {
	 	if (d.Impl() != Implementation::OpenCL)
	 		throw Error("GPUVector: Invalid implementation \'" + ToString(d.Impl()) + 
	 					"\' (should be OpenCL).");
	 }

	inline bool OnGPU() const {return buffer != nullptr;}

	inline void ToGPU(int F = CL_MEM_READ_WRITE) override {
		auto d = static_cast<GPUDevice*>(device);
		if (! OnGPU()) {
			cl::Context& cont = d->Context();
			buffer = new cl::Buffer(cont, F, sizeof(T)*cpuv->size());
			size = cpuv->size();
		}
		else if (size != cpuv->size())
			throw Error("GPUVector::ToGPU: CPU vector size has changed.");
		auto q = d->Queue(qid);
		cl_int enq = q.enqueueWriteBuffer(*buffer, CL_TRUE, 0, sizeof(T)*size, cpuv->data());
		if (enq != CL_SUCCESS) {
			switch(enq) {
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:  
				throw Error("GPUVector::ToGPU(): CL_MEM_OBJECT_ALLOCATION_FAILURE error during copying buffer to the GPU.");
			case CL_INVALID_OPERATION: 
				throw Error("GPUVector::ToGPU(): CL_INVALID_OPERATION error during copying buffer to the GPU.");
			case CL_OUT_OF_RESOURCES:
				throw Error("GPUVector::ToGPU(): CL_OUT_OF_RESOURCES error during copying buffer to the GPU.");
			case CL_OUT_OF_HOST_MEMORY:
				throw Error("GPUVector::ToGPU(): CL_OUT_OF_HOST_MEMORY error during copying buffer to the GPU.");
			default:
			throw Error("GPUVector::ToGPU(): Unknown error during copying buffer to the GPU. Error number: " + std::to_string(enq));
			}
			
		}
	};
	
	inline void FromGPU() override {
		if (! OnGPU()) throw Error("GPUVector::FromGPU: vector is not on the GPU.");
		else if (size != cpuv->size())
			throw Error("GPUVector::FromGPU: CPU vector size has changed.");
		auto d = static_cast<GPUDevice*>(device);
		auto q = d->Queue(qid);
		q.enqueueReadBuffer(*buffer, CL_TRUE, 0, sizeof(T)*size, cpuv->data());
	}

	inline cl::Buffer& GetBuffer() {
		if (! OnGPU()) throw Error("GPUVector::GetBuffer: not on GPU (GPU buffer is not allocated yet).");
		return *buffer;
	}

	using GPUVectorBase<T>::Size;
  
};


#endif // F_AI_GPUVECTOR_HPP_
