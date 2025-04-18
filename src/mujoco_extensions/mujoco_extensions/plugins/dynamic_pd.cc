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

#include "dynamic_pd.h" // Include the simplified header

#include <vector>
#include <memory> // For std::unique_ptr
#include <iostream> // For warnings/debug prints if needed

#include <mujoco/mjplugin.h> // Required for mjPLUGIN_LIB_INIT and registration
#include <mujoco/mujoco.h>


namespace mujoco::plugin::actuator {

// Create: Simplified - only checks for associated actuators and userdata size.
std::unique_ptr<DynamicPD> DynamicPD::Create(const mjModel* m, int instance) {
    // Check if model provides enough userdata slots
    if (m->nuserdata < 2) {
        mju_warning("DynamicPD plugin instance %d requires nuserdata >= 2 (found %d) to read P/D gains.",
                    instance, m->nuserdata);
        // Fail creation if not enough space for gains
        return nullptr;
    }

    // Find all actuators associated with this plugin instance
    std::vector<int> actuators;
    for (int i = 0; i < m->nu; ++i) { // <-- CORRECTED LINE 42: Iterate up to number of actuators
        // Check association using m->actuator_plugin (assuming it holds instance index)
        // This relies on the XML using <plugin instance="instance_name"> actuator type
        if (m->actuator_plugin[i] == instance) {
             // Check if the actuator has the expected settings for this simple plugin
             if (m->actuator_dyntype[i] != mjDYN_NONE) {
                 mju_warning("DynamicPD plugin instance %d, actuator %d ('%s'): "
                             "dyntype should be 'none' for this simplified plugin.",
                             instance, i, mj_id2name(m, mjOBJ_ACTUATOR, i));
                 // Optionally return nullptr to enforce correctness
                 // return nullptr;
             }
             if (m->actuator_actnum[i] != 0) { // <-- CORRECTED LINE 54
                  mju_warning("DynamicPD plugin instance %d, actuator %d ('%s'): "
                             "actdim should be 0 for this simplified plugin.",
                             instance, i, mj_id2name(m, mjOBJ_ACTUATOR, i));
                 // Optionally return nullptr
                 // return nullptr;
             }
             actuators.push_back(i);
        }
    }

    if (actuators.empty()) {
        mju_warning("No actuators found for DynamicPD plugin instance %d", instance);
        return nullptr;
    }

    // Check if model provides enough controls for the setpoint(s)
    // Find the maximum required ctrl index based on trnid
    int max_ctrl_idx = -1;
    for(int act_id : actuators) {
        int ctrl_idx = m->actuator_trnid[act_id * 2];
        if (ctrl_idx > max_ctrl_idx) {
            max_ctrl_idx = ctrl_idx;
        }
    }
    if (max_ctrl_idx < 0) {
         mju_warning("DynamicPD instance %d: Could not find valid control index for actuators.", instance);
         // This case implies no actuator is properly mapped, though actuators list wasn't empty. Should be rare.
         return nullptr;
    }
    if (m->nu <= max_ctrl_idx) {
         mju_warning("DynamicPD instance %d: Actuator requires ctrl index %d, but model nu=%d is too small.",
                     instance, max_ctrl_idx, m->nu);
         // Fail creation if setpoint cannot be read
         return nullptr;
    }


    // std::cout << "[DynamicPD Instance " << instance << "] Create: Found " << actuators.size() << " actuators." << std::endl;
    return std::unique_ptr<DynamicPD>(new DynamicPD(std::move(actuators)));
}

// GetSetpoint: Simplified - reads setpoint from ctrl[trnid] for dyntype=none
mjtNum DynamicPD::GetSetpoint(const mjModel* m, const mjData* d, int actuator_idx) const {
    mjtNum setpoint_signal = 0;
    // Assuming dyntype is always mjDYN_NONE for this simplified version
    int ctrl_idx = m->actuator_trnid[actuator_idx * 2];

    if (ctrl_idx >= 0 && ctrl_idx < m->nu) { // Check nu bounds
        setpoint_signal = d->ctrl[ctrl_idx];
        // Apply control limits if specified
        if (m->actuator_ctrllimited[actuator_idx]) {
            setpoint_signal = mju_clip(setpoint_signal, m->actuator_ctrlrange[2 * actuator_idx],
                                       m->actuator_ctrlrange[2 * actuator_idx + 1]);
        }
    } else {
        // Should not happen if Create() checks succeeded
        // mju_warning("Invalid ctrl_idx %d for actuator %d (nu=%d)", ctrl_idx, actuator_idx, m->nu);
        setpoint_signal = 0; // Default if index is invalid
    }
    return setpoint_signal;
}


// Compute: Simplified - reads P/D from userdata, setpoint from ctrl, calculates PD force
void DynamicPD::Compute(const mjModel* m, mjData* d, int instance) {
    // Assumes Create() checked nuserdata >= 2
    mjtNum p_gain = d->userdata[0];
    mjtNum d_gain = d->userdata[1];

    for (int i = 0; i < actuators_.size(); i++) {
        int actuator_idx = actuators_[i];

        // Get the setpoint signal for this actuator
        mjtNum setpoint = GetSetpoint(m, d, actuator_idx);

        // Calculate position error
        // Note: actuator_length is the current length/position of the transmission
        mjtNum error = setpoint - d->actuator_length[actuator_idx];

        // Calculate velocity error (error derivative)
        // Target velocity is implicitly 0 for position control
        // Note: actuator_velocity is the current velocity of the transmission
        mjtNum error_dot = 0.0 - d->actuator_velocity[actuator_idx];

        // Calculate total force (P term + D term)
        d->actuator_force[actuator_idx] = p_gain * error + d_gain * error_dot;

        // Apply force limits if specified
        if (m->actuator_forcelimited[actuator_idx]) {
            d->actuator_force[actuator_idx] = mju_clip(
                d->actuator_force[actuator_idx],
                m->actuator_forcerange[2 * actuator_idx],
                m->actuator_forcerange[2 * actuator_idx + 1]);
        }
    }
}

// StateSize: Always 0 for this version
int DynamicPD::StateSize(const mjModel* m, int instance) {
    return 0; // No internal plugin state needed in mjData->plugin_state
}

// ActDim: Always 0 for this version
int DynamicPD::ActDim(const mjModel* m, int instance, int actuator_id) {
    return 0; // No activation state needed in mjData->act
}

// RegisterPlugin: Simplified registration
void DynamicPD::RegisterPlugin() {
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);

    plugin.name = "mujoco.plugin.actuator.DynamicGainPD"; // Keep the same name for XML compatibility
    plugin.capabilityflags = mjPLUGIN_ACTUATOR;

    // No XML attributes are read by this simplified plugin version
    plugin.nattribute = 0;
    plugin.attributes = nullptr;

    plugin.nstate = DynamicPD::StateSize; // Assigns StateSize callback (returns 0)

    // No compute_control_dims callback

    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
        // Create checks for nuserdata >= 2
        std::unique_ptr<DynamicPD> pd = DynamicPD::Create(m, instance);
        if (!pd) {
            return -1; // Creation failed
        }
        d->plugin_data[instance] = reinterpret_cast<uintptr_t>(pd.release());
        // Optionally initialize userdata here if needed, e.g., to default gains
        // if (d->userdata && m->nuserdata >=2) {
        //    d->userdata[0] = 1.0; // Default P
        //    d->userdata[1] = 0.1; // Default D
        // }
        return 0; // Success
    };

    plugin.destroy = +[](mjData* d, int instance) {
        delete reinterpret_cast<DynamicPD*>(d->plugin_data[instance]);
        d->plugin_data[instance] = 0;
    };

    plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                        int instance) {
        // Nothing to reset in this simplified version
    };

    // No actuator_act_dot callback needed
    // plugin.actuator_act_dot = nullptr;

    plugin.compute =
        +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
            // Check if the compute request is for the actuator capability
            if (capability_bit & mjPLUGIN_ACTUATOR) {
                 auto* pd = reinterpret_cast<DynamicPD*>(d->plugin_data[instance]);
                 if (pd) {
                    pd->Compute(m, d, instance);
                 }
            }
        };

    // No advance callback needed
    // plugin.advance = nullptr;

    mjp_registerPlugin(&plugin);
}

// Constructor: Simplified
DynamicPD::DynamicPD(std::vector<int> actuators)
    : actuators_(std::move(actuators)) {}


// Ensure plugin auto-registers using mjPLUGIN_LIB_INIT
mjPLUGIN_LIB_INIT {
    // std::cout << "[DynamicPD Simplified] Registering plugin via mjPLUGIN_LIB_INIT..." << std::endl;
    DynamicPD::RegisterPlugin();
    // std::cout << "[DynamicPD Simplified] Plugin registration called." << std::endl;
}

} // namespace mujoco::plugin::actuator
