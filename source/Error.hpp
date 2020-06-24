/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#ifndef INCLUDE_ERROR_HPP_
#define INCLUDE_ERROR_HPP_

#include <string>

class Error {
 public:
  explicit Error(const std::string &s) : error_message(s) {}
  const std::string error_message;
};

#endif  // INCLUDE_ERROR_HPP_
