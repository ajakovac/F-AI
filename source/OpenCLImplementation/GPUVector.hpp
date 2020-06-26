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

	inline void ToGPU() override {
		auto d = static_cast<GPUDevice*>(device);
		if (! OnGPU()) {
			cl::Context& cont = d->Context();
			buffer = new cl::Buffer(cont, CL_MEM_READ_WRITE, sizeof(T)*cpuv->size());
			size = cpuv->size();
		}
		else if (size != cpuv->size())
			throw Error("GPUVector::ToGPU: CPU vector size has changed.");
		auto q = d->Queue(qid);
		q.enqueueWriteBuffer(*buffer, CL_TRUE, 0, sizeof(T)*size, cpuv->data());
	};
	
	inline void FromGPU() override {
		if (! OnGPU()) throw Error("GPUVector::FromGPU: vector is not on the GPU.");
		else if (size != cpuv->size())
			throw Error("GPUVector::FromGPU: CPU vector size has changed.");
		auto d = static_cast<GPUDevice*>(device);
		auto q = d->Queue(qid);
		q.enqueueReadBuffer(*buffer, CL_TRUE, 0, sizeof(T)*size, cpuv->data());
	}

	inline cl::Buffer& GetBuffer() {return *buffer;}
  
};


#endif // F_AI_GPUVECTOR_HPP_
