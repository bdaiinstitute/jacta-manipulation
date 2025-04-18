// Copyright 2025 Google LLC
// Simplified version without slew/dynamics support.
// Based on original PID plugin:
// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_ACTUATOR_SIMPLE_DYNAMIC_PD_H_
#define MUJOCO_PLUGIN_ACTUATOR_SIMPLE_DYNAMIC_PD_H_

#include <vector>
#include <memory> // For std::unique_ptr

#include <mujoco/mujoco.h>

namespace mujoco::plugin::actuator {

// A simple Proportional-Derivative (PD) controller plugin.
// P and D gains are read dynamically from mjData->userdata[0] and userdata[1].
// Setpoint is read from mjData->ctrl[actuator_trnid].
// This version does NOT support slew rate limiting or actuator dynamics.
class DynamicPD {
 public:
  // Creates a new DynamicPD instance based on model information.
  // Returns nullptr on failure (e.g., nuserdata < 2, no associated actuators).
  static std::unique_ptr<DynamicPD> Create(const mjModel* m, int instance);

  // Computes the actuator forces based on the PD control law.
  // Reads P/D gains from d->userdata[0], d->userdata[1].
  // Reads setpoint from d->ctrl[actuator_trnid].
  void Compute(const mjModel* m, mjData* d, int instance);

  // Returns the size of the plugin-specific state in mjData->plugin_state. Always 0.
  static int StateSize(const mjModel* m, int instance);

  // Calculates required activation dimensions (mjData->act) for an actuator. Always 0.
  static int ActDim(const mjModel* m, int instance, int actuator_id);

  // Registers the DynamicPD plugin with MuJoCo.
  static void RegisterPlugin();

  // Constructor (now simpler)
  explicit DynamicPD(std::vector<int> actuators);

 private:
  // Retrieves the Setpoint value from the control signal mapped to the actuator.
  mjtNum GetSetpoint(const mjModel* m, const mjData* d, int actuator_idx) const;

  const std::vector<int> actuators_; // Indices of actuators controlled by this instance
};

}  // namespace mujoco::plugin::actuator

#endif  // MUJOCO_PLUGIN_ACTUATOR_SIMPLE_DYNAMIC_PD_H_
