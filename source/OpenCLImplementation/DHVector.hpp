#ifndef F_AI_DHVECTOR_HPP_
#define F_AI_DHVECTOR_HPP_

#include "GPUVector.hpp"
#include <vector>

template<class T>
class DHVector {
public:

	template<typename A, typename... Args>
	void printA(A&& arg, Args&&... args) {
		std::cout << arg << "\n";
		printA(args...);
	}

	void printA() {}

	template<typename... Args>
	inline DHVector(std::pair<GPUDevice*, std::size_t> const& dandq, Args&&... args) :
		cpuv(std::forward<Args>(args)...), gpuv(cpuv, *dandq.first, dandq.second) { }

	template<typename... Args>
	inline DHVector(GPUDevice* d, Args&&... args) : DHVector<T>({d, 0}, std::forward<Args>(args)...) { }

	// ---------------------------------------------------------------------------------------
    // Getters:

	GPUVector<T>& GPUVec() {return gpuv;}

	inline bool OnGPU() const {return gpuv.OnGPU();}

	inline cl::Buffer& GPUBuffer() {
		if (!OnGPU()) throw Error("DHVector::GPUBuffer: not on GPU (GPU buffer is not allocated yet).");
		return *(gpuv->GetBuffer());
	}

	inline std::size_t Size() const {return gpuv.Size();}
	inline std::size_t size() const {return gpuv.Size();}

	inline std::vector<T>::iterator begin() {return cpuv.begin();}
	inline std::vector<T>::iterator end() {return cpuv.end();}
	inline std::vector<T>::const_iterator begin() const {return cpuv.begin();}
	inline std::vector<T>::const_iterator end() const {return cpuv.end();}
	inline std::vector<T>::const_iterator cbegin() const {return cpuv.cbegin();}
	inline std::vector<T>::const_iterator cend() const {return cpuv.cend();}

	inline T& operator[] (std::size_t n) {return cpuv[n];}
	inline T const& operator[] (std::size_t n) const {return cpuv[n];}
	inline T& back() {return cpuv.back();}
	inline T const& back() const {return cpuv.back();}

	// ---------------------------------------------------------------------------------------
    // Modifiers:

	inline void ToGPU() {
		gpuv.ToGPU();}
	inline void FromGPU() {gpuv.FromGPU();}

private:
	std::vector<T> cpuv;
	GPUVector<T> gpuv;
};

template<typename T>
std::ostream & operator<<(std::ostream &s, const DHVector<T> &x) {
    s << "(";
    if (x.size() > 1) 
    	for ( std::size_t i = 0; i < x.size()-1; ++i) 
    		s << x[i] << ", ";
    if (x.size() > 0) s << x.back();
    s << ")";
    return s;
}

#endif // F_AI_DHVECTOR_HPP_