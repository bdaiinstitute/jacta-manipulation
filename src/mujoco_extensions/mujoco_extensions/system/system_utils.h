// Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
#pragma once

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "mujoco_extensions/system/eigen_types.h"

namespace SystemUtils
{
    using EigenTypes::MatrixT;
    using EigenTypes::MatrixTList;
    using EigenTypes::VectorT;
    using EigenTypes::VectorTList;
    using MatrixRef = Eigen::Ref<MatrixT>;
    using UnalignedMap = Eigen::Map<MatrixT, Eigen::Unaligned>;
    using MapList = std::vector<UnalignedMap>;

    const char kCommandKey[] = "command";
    const char kDefaultKey[] = "default";
    const char kGoalKey[] = "goal";
    const char kGoalScaleKey[] = "distance_scaling";
    const char kLowerBoundKey[] = "lower_bound";
    const char kModelFilename[] = "model_filename";
    const char kPolicyFilename[] = "policy_filename";
    const char kProxScaleKey[] = "proximity_scaling";
    const char kSpotFallPenaltyKey[] = "spot_fall_penalty";
    const char kObjectFallPenaltyKey[] = "object_fall_penalty";
    const char kRewardKey[] = "reward";
    const char kStartKey[] = "start";
    const char kStateKey[] = "state";
    const char kUpperBoundKey[] = "upper_bound";

    /** @brief Loads a Mujoco model from an XML
     *
     * Requires the model_filepath be defined before calling
     *
     * @param model_filepath filepath to the mujoco model
     */
    mjModel *loadModel(const std::string &model_filepath);

    /** @brief Set the state of the MuJoCo model.
     *
     * This function sets the positions and velocities of the MuJoCo model based on the provided state vector.
     *
     * @param model Pointer to the MuJoCo model.
     * @param data Pointer to the MuJoCo data.
     * @param state Vector containing the state (positions and velocities).
     */
    void setState(const mjModel *model, mjData *data, const VectorT &state);

    /** @brief Retrieves the Kp gain from a Mujoco model
     *
     * @param model Mujoco model
     * @returns VectorT containing the Kp gains for the joints of system
     */
    VectorT getJointProportionalGains(const mjModel *model);

    /** @brief Performs a rollout of the physics simulator and updates the internal states of the System
     *
     * @param model pointer to a Mujoco model
     * @param data pointer to a Mujoco data
     * @param state state vector input to the simulator
     * @param control control matrix input to the simulator
     * @returns A tuple of states and sensor values from the rollout steps
     */

    std::tuple<MatrixT, MatrixT> physicsRollout(const mjModel *model, mjData *data, const VectorT &state,
                                                const MatrixT &control);

    /** @brief Performs a rollout on each of the physics simulators stored in the model vector
     *
     * @param model vector of pointers to a Mujoco model
     * @param data vector of pointers to a Mujoco data
     * @param state vector of state vector input to the simulator
     * @param control vector of control matrix input to the simulator
     * @returns A tuple of vectors of states and sensor values from the rollout steps
     */
    std::tuple<MatrixTList, MatrixTList> threadedPhysicsRollout(const std::vector<const mjModel *> &model,
                                                                const std::vector<mjData *> &data, const VectorTList &state,
                                                                const MatrixTList &control);

    /**
     * @brief Performs a single-threaded physics rollout and writes results into preallocated buffers
     *
     * @param model    Pointer to a Mujoco model
     * @param data     Pointer to a Mujoco data struct
     * @param state    Initial state vector of size (nq + nv)
     * @param control  Control matrix of size (horizon x nu)
     * @param states   Output matrix of size (horizon x (nq + nv)), must be preallocated
     * @param sensors  Output matrix of size (horizon x nsensordata), must be preallocated
     */
    void physicsRolloutInPlace(
        const mjModel *model,
        mjData *data,
        const VectorT &state,
        const MatrixT &control,
        MatrixRef states,
        MatrixRef sensors);

    /**
     * @brief Performs a physics rollout on each simulator in parallel and writes results into preallocated buffers
     *
     * @param model     Vector of pointers to Mujoco models, one per thread
     * @param data      Vector of pointers to Mujoco data structs, one per thread
     * @param state     Vector of initial state vectors, one per thread
     * @param control   Vector of control matrices, one per thread
     * @param states    Vector of output state matrices, one per thread; each must be preallocated to (horizon x state_dim)
     * @param sensors   Vector of output sensor matrices, one per thread; each must be preallocated to (horizon x nsensordata)
     */
    void threadedPhysicsRolloutInPlace(
        const std::vector<const mjModel *> &model,
        const std::vector<mjData *> &data,
        const VectorTList &state,
        const MatrixTList &control,
        MapList &states,
        MapList &sensors);

} // namespace SystemUtils
