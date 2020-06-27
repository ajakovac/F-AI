#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <vector>
#include <string>
#include <iostream>

class Timer {
	std::vector<std::chrono::system_clock::time_point> times;
	std::vector<std::string> labels;
	using default_format = std::chrono::nanoseconds;
public:
	Timer() { Record("Start"); }
	std::size_t Record(std::string const& lab) {
		times.push_back(std::chrono::system_clock::now());
		labels.push_back(lab);
		return times.size()-1;
	}
	double sec() const {return sec(times.size()-1); }
	double sec(std::size_t n) const {return dsec(0, n);}
	double dsec(std::size_t n) const {return dsec(n-1, n);}
	double dsec(std::size_t n, std::size_t m) const {
		return (double)(std::chrono::duration_cast<default_format>(times[m] - times[n]).count())*1e-9;
	}

	std::string const& Label(std::size_t n) const {return labels[n];}

	std::size_t Size() const {return times.size();}
};

std::ostream & operator<<(std::ostream &s, const Timer& T) {
	for (std::size_t i = 1; i < T.Size(); ++i)
		s << T.Label(i) << ": " << T.dsec(i) << "\n";
	s << "Total elapsed time: " << T.sec() << "\n";
    return s;
}


#endif