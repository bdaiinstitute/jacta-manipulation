import mujoco
import pytest

# (Import mujoco_extensions and mj_loadAllPluginLibraries as before)
try:
    import mujoco_extensions  # noqa

    print("Imported mujoco_extensions successfully.")
except ImportError:
    pytest.fail("Could not import 'mujoco_extensions'.")


# --- XML for Simplified Plugin (using userdata for gains) ---
XML_PD_USERDATA_BASIC = """
<mujoco model="test_pd_userdata_basic">
  <option timestep="0.002"/>

  <size nuserdata="2"/>

  <extension>
    <plugin plugin="mujoco.plugin.actuator.DynamicGainPD">
       <instance name="pd_instance">
         </instance>
    </plugin>
  </extension>

  <worldbody>
    <body name="box" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <joint name="slider" type="slide" axis="1 0 0" damping="0.1"/>
    </body>
  </worldbody>

  <actuator>
    <plugin name="actuator0" joint="slider"
            plugin="mujoco.plugin.actuator.DynamicGainPD" instance="pd_instance"
            ctrlrange="-100 100"
            actdim="0" dyntype="none" /> </actuator>

  <sensor>
      <actuatorfrc name="force_sensor" actuator="actuator0"/>
      <jointpos name="pos_sensor" joint="slider"/>
      <jointvel name="vel_sensor" joint="slider"/>
  </sensor>

</mujoco>
"""


# --- Test 1: Loading ---
def test_load_minimal_model_userdata():
    """Tests loading the simplified model using userdata."""
    print("\n--- test_load_minimal_model_userdata ---")
    try:
        model = mujoco.MjModel.from_xml_string(XML_PD_USERDATA_BASIC)
        print("SUCCESS: Minimal userdata model loaded.")
        assert model is not None, "Model loading returned None"
        assert model.nplugin == 1, f"Expected 1 plugin instance, found {model.nplugin}"
        # Check if enough userdata was allocated (could be more than 2)
        assert model.nuserdata >= 2, f"Expected nuserdata>=2, found {model.nuserdata}"
        # Check if control dimension is correct (should be 1 for the setpoint)
        assert model.nu == 1, f"Expected nu=1, found {model.nu}"
    except Exception as e:
        pytest.fail(
            f"Failed to load minimal userdata XML model: {type(e).__name__}: {e}",
            pytrace=True,
        )


# --- Test 2: Stepping ---
def test_step_minimal_model_userdata():
    """Tests stepping the simulation with userdata gains."""
    print("\n--- test_step_minimal_model_userdata ---")
    try:
        print("Loading model...")
        model = mujoco.MjModel.from_xml_string(XML_PD_USERDATA_BASIC)
        data = mujoco.MjData(model)
        print(
            f"Model loaded (nu={model.nu}, nuserdata={model.nuserdata}), Data created."
        )
        assert model.nu == 1, "Model nu should be 1"
        assert model.nuserdata >= 2, "Model nuserdata should be >= 2"

        # Set initial state
        data.qpos[0] = 0.1
        data.qvel[0] = -0.1
        print(f"Initial state: qpos={data.qpos[0]}, qvel={data.qvel[0]}")

        # Set P/D gains in userdata
        data.userdata[0] = 10.0  # P gain
        data.userdata[1] = 1.0  # D gain
        print(f"Set userdata: P={data.userdata[0]}, D={data.userdata[1]}")

        # Set setpoint in ctrl (only ctrl[0] exists)
        data.ctrl[0] = 0.5  # Setpoint
        print(f"Set ctrl: Setpoint={data.ctrl[0]}")

        # Perform one simulation step
        print("Calling mj_step...")
        mujoco.mj_step(model, data)
        print("SUCCESS: mj_step completed without crashing.")

        # Quick check: state should ideally have changed
        print(f"State after step: qpos={data.qpos[0]:.4f}, qvel={data.qvel[0]:.4f}")
        assert (
            data.qpos[0] != 0.1 or data.qvel[0] != -0.1
        ), "State did not change after step"

    except Exception as e:
        pytest.fail(
            f"Test failed during execution: {type(e).__name__}: {e}", pytrace=True
        )


# --- Test 3: Basic PD Force Calculation ---
@pytest.mark.parametrize(
    "p_gain, d_gain, setpoint, initial_pos, initial_vel",
    [
        (10.0, 1.0, 0.5, 0.0, 0.0),
        (5.0, 0.0, -0.2, 0.1, 0.0),
        (0.0, 2.0, 0.0, 0.0, 0.5),
        (20.0, 3.0, 0.0, -0.1, -0.2),
        (15.0, 0.5, 0.3, 0.3, 0.1),
    ],
)
def test_pd_force_calculation_userdata(
    p_gain, d_gain, setpoint, initial_pos, initial_vel
):
    """Tests the basic PD force calculation using userdata for gains."""
    print(
        f"\n--- test_pd_force_calculation_userdata [{p_gain=}, {d_gain=}, {setpoint=}, {initial_pos=}, {initial_vel=}] ---"
    )
    try:
        # timestep from XML is 0.002
        model = mujoco.MjModel.from_xml_string(XML_PD_USERDATA_BASIC)
        data = mujoco.MjData(model)
        print(
            f"Model loaded (nu={model.nu}, nuserdata={model.nuserdata}), Data created."
        )
        assert model.nu == 1, "Model nu should be 1"
        assert model.nuserdata >= 2, "Model nuserdata should be >= 2"

        # Find sensor IDs
        force_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor"
        )
        pos_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "pos_sensor"
        )
        vel_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "vel_sensor"
        )
        assert all(
            id >= 0 for id in [force_sensor_id, pos_sensor_id, vel_sensor_id]
        ), "Sensor not found"

        # Set initial conditions
        data.qpos[0] = initial_pos
        data.qvel[0] = initial_vel
        print(f"Set initial state: qpos={data.qpos[0]}, qvel={data.qvel[0]}")

        # Set gains in userdata
        data.userdata[0] = p_gain
        data.userdata[1] = d_gain
        print(f"Set userdata: P={data.userdata[0]}, D={data.userdata[1]}")

        # Set setpoint in ctrl
        data.ctrl[0] = setpoint
        print(f"Set ctrl: Setpoint={data.ctrl[0]}")

        # Call mj_forward to update sensors based on initial state BEFORE stepping
        print("Calling mj_forward...")
        mujoco.mj_forward(model, data)

        # Read state from sensors *before* the step (values used for force calculation)
        pos_at_start = data.sensordata[pos_sensor_id]
        vel_at_start = data.sensordata[vel_sensor_id]
        print(f"State at start of step: pos={pos_at_start:.4f}, vel={vel_at_start:.4f}")

        # Calculate expected force based on state at start of step
        pos_error = setpoint - pos_at_start
        vel_error = 0.0 - vel_at_start  # Target velocity is implicitly 0
        expected_force = p_gain * pos_error + d_gain * vel_error
        print(f"Errors: pos_err={pos_error:.4f}, vel_err={vel_error:.4f}")
        print(f"Expected Force: {expected_force:.6f}")

        # Perform one simulation step (this computes and applies the force)
        print("Calling mj_step...")
        mujoco.mj_step(model, data)
        print("mj_step completed.")

        # Read the actual force applied during the step from the sensor
        actual_force = data.sensordata[force_sensor_id]
        print(f"Actual Force (from sensor): {actual_force:.6f}")

        # Compare actual vs expected force
        assert actual_force == pytest.approx(expected_force, abs=1e-6)
        print("SUCCESS: Actual force matches expected force.")

    except Exception as e:
        pytest.fail(
            f"Test failed during execution: {type(e).__name__}: {e}", pytrace=True
        )
