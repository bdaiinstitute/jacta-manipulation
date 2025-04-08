// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "mujoco_extensions/onnx_interface/onnx_interface.h"

namespace OnnxInterface
{

  Policy::Policy() : input_tensor_(nullptr), memory_info_(nullptr), run_options_(nullptr)
  {
    session = nullptr;
  }

  Policy::Policy(const std::string &policy_path_)
      : policy_path(policy_path_), input_tensor_(nullptr), memory_info_(nullptr), run_options_(nullptr)
  {
    // Creates and tracks the ONNX session
    session = OnnxInterface::allocateOrtSession(policy_path);
    initializePolicy();
  }

  Policy::Policy(const std::string &policy_path_, std::shared_ptr<Ort::Session> session_)
      : policy_path(policy_path_), input_tensor_(nullptr), memory_info_(nullptr), run_options_(nullptr)
  {
    // Only copies the pointer to the specified session
    session = std::move(session_);
    initializePolicy();
  }

  Policy &Policy::operator=(const Policy &policy)
  {
    session = std::move(policy.session);
    policy_path = policy.policy_path;
    initializePolicy();
    return *this;
  }

  Policy::~Policy() {}

  void Policy::initializePolicy()
  {
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Allocator allocator = Ort::Allocator(*session, memory_info_);
    num_input_nodes = session->GetInputCount();

    // Creates the input name and nodes for us to work with
    // Here we assume the session/policy file only has a single input and output
    input_name_ = session->GetInputNameAllocated(0, allocator).get();
    input_shape_ = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_size = input_shape_[1];

    output_name_ = session->GetOutputNameAllocated(0, allocator).get();
    output_shape_ = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    output_size = output_shape_[1];

    input_tensors.emplace_back(createZeroTensor(input_size, input_shape_));
    output_tensors.emplace_back(createZeroTensor(output_size, output_shape_));
  }

  std::string Policy::getInputName()
  {
    return input_name_;
  }

  std::string Policy::getOutputName()
  {
    return output_name_;
  }

  std::vector<int64_t> Policy::getInputShape()
  {
    return input_shape_;
  }

  std::vector<int64_t> Policy::getOutputShape()
  {
    return output_shape_;
  }

  Ort::Value Policy::createZeroTensor(int64_t tensor_length, const std::vector<int64_t> &tensor_shape)
  {
    VectorT placeholder_(tensor_length);
    placeholder_.setZero();
    return createTensorFromVector(&placeholder_, tensor_shape);
  }

  Ort::Value Policy::createTensorFromVector(VectorT *input_vector, const std::vector<int64_t> &vector_shape)
  {
    return Ort::Value::CreateTensor<float>(memory_info_, input_vector->data(), input_vector->size(), vector_shape.data(),
                                           vector_shape.size());
  }

  VectorT Policy::policyInference(VectorT *input_vector)
  {
    // Grab C‑strings from your persistent std::string members
    const char *input_name_c = input_name_.c_str();
    const char *output_name_c = output_name_.c_str();

    // Build name arrays (1 element each)
    std::array<const char *, 1> input_names = {input_name_c};
    std::array<const char *, 1> output_names = {output_name_c};

    // Create a fresh input tensor from your vector
    Ort::Value new_input_tensor = createTensorFromVector(input_vector, input_shape_);
    input_tensors[0] = std::move(new_input_tensor);

    // Run the session
    auto results = session->Run(
        run_options_,
        input_names.data(), // inputs
        input_tensors.data(),
        static_cast<int>(input_tensors.size()),
        output_names.data(),  // outputs
        output_tensors.size() // note: this should be the number of outputs
    );

    // Map the first (and only) output back into an Eigen vector
    float *out_data = results[0].GetTensorMutableData<float>();
    return Eigen::Map<VectorT>(out_data, output_size);
  }

  std::shared_ptr<Ort::Session> allocateOrtSession(const std::string &policy_filepath)
  {
    Ort::Env env;
    Ort::SessionOptions session_options;
    // TODO(@bhung) explain why you need to restrict threads
    session_options.SetIntraOpNumThreads(1);
    return std::make_shared<Ort::Session>(env, policy_filepath.c_str(), session_options);
  }

  std::shared_ptr<Ort::Session> getNullSession()
  {
    return std::shared_ptr<Ort::Session>(nullptr);
  }

} // namespace OnnxInterface
