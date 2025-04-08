// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#include <chrono>

#include "mujoco_extensions/system/system_class.h"

namespace SystemClass
{

  // Constructor
  System::System(const std::string &model_filepath_, const std::string &policy_filepath_)
      : model_filepath(model_filepath_),
        policy_filepath(policy_filepath_),
        observation(3 + 3 + 3 + 3 + 7 + 12 + 3 + 19 + 19 + 12), // 84
        policy_output(1),
        policy_input_(1),
        control_(19),
        default_joint_pos_(19),
        joint_pos_(19),
        joint_vel_(19)
  {
    // Initialize the permutation matrices
    initializeSystemIndices();
    zeroOutVectors();

    model = SystemUtils::loadModel(model_filepath);
    data = mj_makeData(model);
    setStateIndices();
    loadPolicy(policy_filepath, OnnxInterface::getNullSession());

    policy_input_.resize(policy.input_size);
    policy_output.resize(policy.output_size);
  }

  System::System(const std::string &model_filepath_, const std::string &policy_filepath_, const mjModel *reference_model,
                 std::shared_ptr<OnnxInterface::Session> reference_session)
      : model_filepath(model_filepath_),
        policy_filepath(policy_filepath_),
        observation(3 + 3 + 3 + 3 + 7 + 12 + 3 + 19 + 19 + 12), // 84
        policy_output(1),
        policy_input_(1),
        control_(19),
        default_joint_pos_(19),
        joint_pos_(19),
        joint_vel_(19)
  {
    // Initialize the permutation matrices
    initializeSystemIndices();
    zeroOutVectors();

    model = mj_copyModel(nullptr, reference_model);
    if (!model)
    {
      throw std::runtime_error("Failed to load copy XML file");
    }
    data = mj_makeData(model);
    setStateIndices();

    loadPolicy(policy_filepath, reference_session);
    policy_input_.resize(policy.input_size);
    policy_output.resize(policy.output_size);
  }

  System::~System()
  {
    if (data)
    {
      mj_deleteData(data);
    }
    if (model)
    {
      mj_deleteModel(model);
    }
  }

  void System::reset(const bool reset_last_output)
  {
    mj_resetData(model, data);
    if (reset_last_output)
    {
      policy_output.setZero();
    }
    for (int i = 0; i < model->nsensordata; i++)
    {
      data->sensordata[i] = 0.0;
    }
  }

  void System::zeroOutVectors()
  {
    observation.setZero();
    control_.setZero();
    policy_output.setZero();
    joint_pos_.setZero();
    joint_vel_.setZero();
  }

  void System::setStateIndices()
  {
    int base_id = mj_name2id(model, mjOBJ_BODY, "body");                       // id of the base
    base_qpos_start_idx = model->jnt_qposadr[model->body_jntadr[base_id]];     // base position address
    base_qvel_start_idx = model->jnt_dofadr[model->body_jntadr[base_id]];      // base velocity address
    int first_leg_id = mj_name2id(model, mjOBJ_BODY, "front_left_hip");        // id of the first leg
    leg_qpos_start_idx = model->jnt_qposadr[model->body_jntadr[first_leg_id]]; // leg position address
    leg_qvel_start_idx = model->jnt_dofadr[model->body_jntadr[first_leg_id]];  // leg velocity address
  }

  void System::loadPolicy(const std::string &policy_filepath_,
                          std::shared_ptr<OnnxInterface::Session> reference_session)
  {
    if (reference_session.get() == nullptr)
    {
      policy = OnnxInterface::Policy(policy_filepath_);
    }
    else
    {
      policy = OnnxInterface::Policy(policy_filepath_, reference_session);
    }
  }

  void System::initializeSystemIndices()
  {
    Eigen::ArrayXi orbit_to_mujoco_legs_indices(12);
    orbit_to_mujoco_legs_indices << 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11;
    orbit_to_mujoco_legs.indices() = orbit_to_mujoco_legs_indices;

    Eigen::ArrayXi mujoco_to_orbit_legs_indices(12);
    mujoco_to_orbit_legs_indices << 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11;
    mujoco_to_orbit_legs.indices() = mujoco_to_orbit_legs_indices;

    Eigen::ArrayXi orbit_to_mujoco_indices(19);
    orbit_to_mujoco_indices << 12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18;
    orbit_to_mujoco.indices() = orbit_to_mujoco_indices;

    Eigen::ArrayXi mujoco_to_orbit_indices(19);
    mujoco_to_orbit_indices << 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18;
    mujoco_to_orbit.indices() = mujoco_to_orbit_indices;

    default_joint_pos_ << 0.12, 0.5, -1, -0.12, 0.5, -1, 0.12, 0.5, -1, -0.12, 0.5, -1, // legs pos (mujoco)
        0, -0.9, 1.8, 0, -0.9, 0, -1.54;                                                // arm pos (mujoco)
  }

  // Compute observation
  void System::setObservation(const VectorT &command)
  {
    VectorT torso_vel_command = command.segment(0, 3);
    VectorT arm_command = command.segment(3, 7);
    VectorT leg_command = command.segment(10, 12);
    VectorT torso_pos_command = command.segment(22, 3);

    // Initialization
    int offset = 0;
    double inv_base_quat[4];
    double base_linear_velocity[3];
    double base_angular_velocity[3];
    double projected_gravity[3] = {0, 0, -1.0};

    // Indices
    const int base_quat_start_index = this->base_qpos_start_idx + 3;
    const int base_linvel_start_index = this->base_qvel_start_idx;
    const int base_angvel_start_index = this->base_qvel_start_idx + 3;
    const int joint_pos_start_index = this->leg_qpos_start_idx;
    const int joint_vel_start_index = this->leg_qvel_start_idx;

    // Compute observation data
    mju_negQuat(inv_base_quat, data->qpos + base_quat_start_index);

    mju_rotVecQuat(base_linear_velocity, data->qvel + base_linvel_start_index, inv_base_quat);

    for (int i = 0; i < 3; ++i)
    {
      base_angular_velocity[i] = data->qvel[base_angvel_start_index + i];
    }

    mju_rotVecQuat(projected_gravity, projected_gravity, inv_base_quat);

    for (int i = 0; i < 19; i++)
    {
      joint_pos_[i] = data->qpos[joint_pos_start_index + i] - default_joint_pos_[i];
      joint_vel_[i] = data->qvel[joint_vel_start_index + i];
    }
    joint_pos_ = mujoco_to_orbit * joint_pos_;
    joint_vel_ = mujoco_to_orbit * joint_vel_;

    // Populate the observation vector
    for (int i = 0; i < 3; i++)
    {
      observation[offset + i] = base_linear_velocity[i];
    }
    offset += 3;

    for (int i = 0; i < 3; i++)
    {
      observation[offset + i] = base_angular_velocity[i];
    }
    offset += 3;

    for (int i = 0; i < 3; i++)
    {
      observation[offset + i] = projected_gravity[i];
    }
    offset += 3;

    for (int i = 0; i < 3; i++)
    {
      observation[offset + i] = torso_vel_command[i];
    }
    offset += 3;
    for (int i = 0; i < 7; i++)
    {
      observation[offset + i] = arm_command[i];
    }
    offset += 7;

    for (int i = 0; i < 12; i++)
    {
      observation[offset + i] = leg_command[i];
    }
    offset += 12;

    for (int i = 0; i < 3; i++)
    {
      observation[offset + i] = torso_pos_command[i];
    }
    offset += 3;

    for (int i = 0; i < 19; i++)
    {
      observation[offset + i] = joint_pos_[i];
    }
    offset += 19;

    for (int i = 0; i < 19; i++)
    {
      observation[offset + i] = joint_vel_[i];
    }
    offset += 19;

    for (int i = 0; i < 12; i++)
    {
      observation[offset + i] = policy_output[i];
    }
    offset += 12;
  }

  // evaluate neural network policy
  void System::policyInference()
  {
    // TODO(@bhung) Get most of these hard coded index values from another source
    // Convert observation to float and initialize torch_input
    for (int i = 0; i < this->policy.input_size; ++i)
    {
      this->policy_input_[i] = static_cast<float>(this->observation[i]);
    }
    OnnxInterface::VectorT policy_input_float_ = policy_input_.cast<float>();
    OnnxInterface::VectorT policy_output_float_ = this->policy.policyInference(&policy_input_float_);
    this->policy_output = policy_output_float_.cast<double>();

    // legs (output by the neural net)
    control_.segment(0, 12) = 0.2 * policy_output;
    control_.segment(0, 12) = orbit_to_mujoco_legs * control_.segment(0, 12);
    control_.segment(0, 12) = default_joint_pos_.segment(0, 12) + control_.segment(0, 12);
    // arm (read neural net input i.e. from observation)
    control_.segment(12, 7) = observation.segment(3 + 3 + 3 + 3, 7);
    // overwrite leg joint positions (read neural net input i.e. from observation)
    VectorT leg_joint_command = observation.segment(3 + 3 + 3 + 3 + 7, 12);
    if (leg_joint_command.segment(0, 3).norm() > 0)
    { // FL
      control_.segment(0, 3) = leg_joint_command.segment(0, 3);
    }
    else if (leg_joint_command.segment(3, 3).norm() > 0)
    { // FR
      control_.segment(3, 3) = leg_joint_command.segment(3, 3);
    }
    else if (leg_joint_command.segment(6, 3).norm() > 0)
    { // HL
      control_.segment(6, 3) = leg_joint_command.segment(6, 3);
    }
    else if (leg_joint_command.segment(9, 3).norm() > 0)
    { // HR
      control_.segment(9, 3) = leg_joint_command.segment(9, 3);
    }

    for (int i = 0; i < 19; i++)
    {
      data->ctrl[i] = control_[i];
    }
  }

  // extract the control vector out of the mj Data
  VectorT System::getControl()
  {
    const int nu = model->nu;
    VectorT control(nu);
    for (int i = 0; i < nu; i++)
    {
      control[i] = data->ctrl[i];
    }
    return control;
  }

  // extracts the state vector out of the mj model
  VectorT System::getState()
  {
    int nq = model->nq;
    int nv = model->nv;
    VectorT state_vec(nq + nv);
    for (int i = 0; i < nq; ++i)
    {
      state_vec[i] = data->qpos[i];
    }
    for (int i = 0; i < nv; ++i)
    {
      state_vec[nq + i] = data->qvel[i];
    }
    return state_vec;
  }

  std::tuple<MatrixT, MatrixT> System::rollout(const VectorT &state, const MatrixT &command, const int physics_substeps,
                                               const bool reset_last_output, const double cutoff_time)
  {
    // Resetting mujoco data, last output and sensordata to zero
    this->reset(reset_last_output);

    // Initialization
    const auto num_commands = command.rows();
    const auto num_steps = num_commands * physics_substeps;
    const int nq = model->nq;
    const int nv = model->nv;
    const int nsensordata = model->nsensordata;
    MatrixT states = MatrixT::Zero(num_steps, nq + nv);
    MatrixT sensors = MatrixT::Zero(num_steps, nsensordata);
    SystemUtils::setState(model, data, state);

    // Rollout physics and policy
    int ctr = 0;
    double seconds_elapsed = 0.0;
    auto rollout_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_commands; ++i)
    {
      for (int k = 0; k < physics_substeps; ++k)
      {
        // While below the cutoff time, calculate the mujoco steps
        auto loop_start_time = std::chrono::high_resolution_clock::now();
        seconds_elapsed = std::chrono::duration<double>(loop_start_time - rollout_start_time).count();
        if (seconds_elapsed < cutoff_time)
        {
          this->setObservation(command.row(i));
          this->System::policyInference();
          mj_step(model, data);
          for (int l = 0; l < nq; ++l)
          {
            states(ctr, l) = data->qpos[l];
          }
          for (int l = 0; l < nv; ++l)
          {
            states(ctr, nq + l) = data->qvel[l];
          }
          for (int l = 0; l < nsensordata; ++l)
          {
            sensors(ctr, l) = data->sensordata[l];
          }
        }
        else
        {
          // Fill in the rest of the state with the last calculated state and stop the rollout
          int prev_ind = std::max(ctr - 1, 0);
          for (int l = 0; l < nq; ++l)
          {
            states(ctr, l) = states(prev_ind, l);
          }
          for (int l = 0; l < nv; ++l)
          {
            states(ctr, nq + l) = states(prev_ind, nq + l);
          }
          for (int l = 0; l < nsensordata; ++l)
          {
            sensors(ctr, l) = sensors(prev_ind, l);
          }
        }
        ctr += 1;
      }
    }

    return {states, sensors};
  }

  std::tuple<MatrixT, MatrixT> System::rollout_world_frame(const VectorT &state, const MatrixT &command,
                                                           const int physics_substeps)
  {
    // Resetting mujoco data, last output and sensordata to zero
    this->reset();

    // Initialization
    const auto num_commands = command.rows();
    const auto num_steps = num_commands * physics_substeps;
    const int nq = model->nq;
    const int nv = model->nv;
    const int nsensordata = model->nsensordata;
    const int base_quat_start_index = this->base_qpos_start_idx + 3;
    MatrixT states = MatrixT::Zero(num_steps, nq + nv);
    MatrixT sensors = MatrixT::Zero(num_steps, nsensordata);
    SystemUtils::setState(model, data, state);
    VectorT transformed_command(command.cols());

    // Rollout physics and policy
    int ctr = 0;
    for (int i = 0; i < num_commands; ++i)
    {
      for (int k = 0; k < physics_substeps; ++k)
      {
        if (ctr % physics_substeps == 0)
        {
          // transform base command to robot frame
          VectorT base_command = command.row(i).segment(0, 3);
          double inv_base_quat[4];
          mju_negQuat(inv_base_quat, data->qpos + base_quat_start_index);
          mju_rotVecQuat(base_command.data(), base_command.data(), inv_base_quat);
          // Reconstruct the full command vector with the transformed base command
          transformed_command.segment(0, 3) = base_command;
          transformed_command.segment(3, command.cols() - 3) = command.row(i).segment(3, command.cols() - 3);

          this->setObservation(transformed_command); // command in robot frame
          this->System::policyInference();
        }
        mj_step(model, data);
        for (int l = 0; l < nq; ++l)
        {
          states(ctr, l) = data->qpos[l];
        }
        for (int l = 0; l < nv; ++l)
        {
          states(ctr, nq + l) = data->qvel[l];
        }
        for (int l = 0; l < nsensordata; ++l)
        {
          sensors(ctr, l) = data->sensordata[l];
        }
        ctr += 1;
      }
    }
    return {states, sensors};
  }

  VectorT System::stateToXYTheta(const VectorT &state)
  {
    VectorT xytheta(3);
    xytheta[0] = state[this->base_qpos_start_idx];
    xytheta[1] = state[this->base_qpos_start_idx + 1];
    double q_w = state[this->base_qpos_start_idx + 3];
    double q_x = state[this->base_qpos_start_idx + 4];
    double q_y = state[this->base_qpos_start_idx + 5];
    double q_z = state[this->base_qpos_start_idx + 6];
    xytheta[2] = atan2(2 * (q_z * q_w + q_x * q_y), 1 - 2 * (q_y * q_y + q_z * q_z));
    return xytheta;
  }

  std::tuple<MatrixT, MatrixT, MatrixT> System::rollout_world_frame_feedback(const VectorT &state, const MatrixT &command,
                                                                             const MatrixT &posref,
                                                                             const VectorT &p_gains,
                                                                             const int physics_substeps)
  {
    // Resetting mujoco data, last output and sensordata to zero
    this->reset();

    // Initialization
    const auto num_commands = command.rows();
    const auto num_steps = num_commands * physics_substeps;
    const int nq = model->nq;
    const int nv = model->nv;
    const int nsensordata = model->nsensordata;
    const int base_quat_start_index = this->base_qpos_start_idx + 3;
    MatrixT states = MatrixT::Zero(num_steps, nq + nv);
    MatrixT sensors = MatrixT::Zero(num_steps, nsensordata);
    MatrixT closed_loop_ctrls = MatrixT::Zero(num_steps, command.cols());
    VectorT transformed_command(command.cols());

    SystemUtils::setState(model, data, state);

    // Rollout physics and policy
    int ctr = 0;

    VectorT current_base_state = this->stateToXYTheta(state);

    for (int i = 0; i < num_commands; ++i)
    {
      for (int k = 0; k < physics_substeps; ++k)
      {
        if (ctr % physics_substeps == 0)
        {
          // transform base command to robot frame
          VectorT base_command = command.row(i).segment(0, 3);
          VectorT posref_row = posref.row(i);
          VectorT control_error = current_base_state - posref_row;
          VectorT gain = p_gains.cwiseProduct(control_error);
          base_command = base_command - gain;
          double inv_base_quat[4];
          VectorT base_command_new(3);
          mju_negQuat(inv_base_quat, data->qpos + base_quat_start_index);
          mju_rotVecQuat(base_command_new.data(), base_command.data(), inv_base_quat);

          // Reconstruct the full command vector with the transformed base command
          transformed_command.segment(0, 2) = base_command_new.segment(0, 2);
          transformed_command[2] = base_command[2];
          transformed_command.segment(3, command.cols() - 3) = command.row(i).segment(3, command.cols() - 3);
          closed_loop_ctrls.row(ctr) = transformed_command; // Store the transformed command

          this->setObservation(transformed_command); // command in robot frame
          this->System::policyInference();
        }
        mj_step(model, data);
        for (int l = 0; l < nq; ++l)
        {
          states(ctr, l) = data->qpos[l];
        }
        for (int l = 0; l < nv; ++l)
        {
          states(ctr, nq + l) = data->qvel[l];
        }
        for (int l = 0; l < nsensordata; ++l)
        {
          sensors(ctr, l) = data->sensordata[l];
        }
        current_base_state = this->stateToXYTheta(states.row(ctr));
        ctr += 1;
      }
    }
    return {states, sensors, closed_loop_ctrls};
  }

  std::tuple<MatrixTList, MatrixTList, VectorTList> threadedRollout(const std::vector<std::shared_ptr<System>> &systems,
                                                                    const VectorTList &states, const MatrixTList &command,
                                                                    const VectorTList &last_policy_outputs,
                                                                    const int num_threads, const int physics_substeps,
                                                                    const double cutoff_time)
  {
    int effective_threads = std::min(static_cast<int>(systems.size()), num_threads);

    // Vector to store threads
    std::vector<std::thread> threads;
    MatrixTList output_states(effective_threads);
    MatrixTList sensors(effective_threads);
    VectorTList policy_outputs(effective_threads);

    // Start threads
    for (int i = 0; i < effective_threads; ++i)
    {
      threads.emplace_back([i, &systems, &states, &command, &last_policy_outputs, &output_states, &sensors,
                            physics_substeps, cutoff_time]()
                           {
      systems[i]->policy_output = last_policy_outputs[i];
      std::tie(output_states[i], sensors[i]) =
          systems[i]->rollout(states[i], command[i], physics_substeps, false, cutoff_time); });
    }

    // Join threads
    for (auto &thread : threads)
    {
      thread.join();
    }

    // Update last_policy_outputs after all threads have completed
    for (int i = 0; i < effective_threads; ++i)
    {
      policy_outputs[i] = systems[i]->policy_output;
    }

    return {output_states, sensors, policy_outputs};
  }

  std::tuple<MatrixTList, MatrixTList, MatrixTList, VectorTList> threadedRolloutFeedbackWorldFrame(
      const std::vector<std::shared_ptr<System>> &systems, const VectorTList &states, const MatrixTList &command,
      const MatrixTList &posref, const VectorT &p_gains, const VectorTList &last_policy_outputs,
      const int physics_substeps, const int num_threads)
  {
    int effective_threads = std::min(static_cast<int>(systems.size()), num_threads);
    MatrixTList output_states(effective_threads);
    MatrixTList sensors(effective_threads);
    MatrixTList closed_loop_ctrls(effective_threads);
    VectorTList policy_outputs(effective_threads);

    // Start threads
    std::vector<std::thread> threads;
    for (int i = 0; i < effective_threads; ++i)
    {
      threads.emplace_back([i, &systems, &states, &command, &posref, &p_gains, &last_policy_outputs, physics_substeps,
                            &output_states, &sensors, &closed_loop_ctrls]()
                           {
      systems[i]->policy_output = last_policy_outputs[i];
      std::tie(output_states[i], sensors[i], closed_loop_ctrls[i]) =
          systems[i]->rollout_world_frame_feedback(states[i], command[i], posref[i], p_gains, physics_substeps); });
    }
    // Join threads
    for (auto &thread : threads)
    {
      thread.join();
    }

    // Update last_policy_outputs after all threads have completed
    for (int i = 0; i < effective_threads; ++i)
    {
      policy_outputs[i] = systems[i]->policy_output;
    }

    return {output_states, sensors, closed_loop_ctrls, policy_outputs};
  }

  std::vector<std::shared_ptr<SystemClass::System>> create_systems_vector(const std::string &model_filepath,
                                                                          const std::string &policy_filepath,
                                                                          const int num_systems)
  {
    // Vector to store the systems
    std::vector<std::shared_ptr<SystemClass::System>> systems(num_systems);
    char error_buffer[1000] = "Could not load XML file";

    // Attempt to load the model from the XML file
    const mjModel *reference_model = mj_loadXML(model_filepath.c_str(), nullptr, error_buffer, sizeof(error_buffer));
    if (!reference_model)
    {
      throw std::runtime_error("Failed to load model XML file: " + model_filepath + " - " + error_buffer);
    }

    std::shared_ptr<OnnxInterface::Session> reference_session =
        std::shared_ptr<OnnxInterface::Session>(OnnxInterface::allocateOrtSession(policy_filepath));
    // Function to create a system in a separate thread
    auto create_system = [&](int i)
    {
      systems[i] =
          std::make_shared<SystemClass::System>(model_filepath, policy_filepath, reference_model, reference_session);
    };

    // Vector to hold the threads
    std::vector<std::thread> threads;
    threads.reserve(num_systems);

    // Launch threads to create System instances in parallel
    for (int i = 0; i < num_systems; i++)
    {
      threads.emplace_back(create_system, i);
    }

    // Wait for all threads to finish
    for (auto &thread : threads)
    {
      thread.join();
    }

    return systems;
  }

} // namespace SystemClass
