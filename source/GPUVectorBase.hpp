#ifndef F_AI_GPUVECTORBASE_HPP_
#define F_AI_GPUVECTORBASE_HPP_

#include <vector>
#include "GPUDeviceBase.hpp"

template<class T>
class GPUVectorBase {
protected:
	std::vector<T>* cpuv;
	std::size_t size;
	GPUDeviceBase* device;
public: 

	inline GPUVectorBase(std::vector<T>& cpu_vec, GPUDeviceBase& d) : cpuv(&cpu_vec), 
		size(cpu_vec.size()), device(&d) {}

	virtual void ToGPU() = 0;
	virtual void FromGPU() = 0;

	inline virtual std::size_t Size() const final {return size;}
};

#endif // F_AI_GPUVECTORBASE_HPP_