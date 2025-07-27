#!/usr/bin/env python3
from rlenv.cassiemujoco import *
import numpy as np
import time
import os

class CassieVisualizer:
    def __init__(self):
        # Get the path to cassie.xml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "assets/cassie.xml")
        
        # Initialize simulation and visualization
        self.sim = CassieSim(model_path)
        self.vis = CassieVis(self.sim, model_path)

        # Initialize state
        self.state = CassieState()
        
        # Initialize counters and timing variables
        self.step_counter = 0
        self.sim_freq = 2000  # Cassie sim runs at 2000Hz
        self.vis_freq = 80    # Desired visualization frequency
        self.steps_per_frame = self.sim_freq // self.vis_freq
        self.dt = 1.0 / self.sim_freq

        self.slow_motion_factor = 10.0
        
        # Set initial pose
        qpos = np.zeros(35)
        qpos[2] = 1.01  # Height
        qpos[3] = 1.0   # Quaternion w
        qpos[7] = 0.0045  # Left hip roll
        qpos[8] = 0.0    # Left hip yaw
        qpos[9] = 0.4973  # Left hip pitch
        qpos[10] = 0.9785  # Left knee
        qpos[21] = -0.0045  # Right hip roll
        qpos[22] = 0.0    # Right hip yaw
        qpos[23] = 0.4973  # Right hip pitch
        qpos[24] = 0.9786  # Right knee
        
        # Set the initial pose
        self.sim.set_qpos(qpos)

        # Set camera view
        self.vis.set_cam("cassie-pelvis",  # target body to track
                        zoom=3.0,          # zoom level (higher = closer)
                        azimuth=90,        # side view (90 = view from right)
                        elevation=-10)      # slightly looking down
        # Initialize PD controller for thrusters
        self.u = pd_in_t()

    def apply_thruster_forces(self, left_force=100.0, right_force=100.0):
        """Apply thruster forces through actuators"""
        # Ensure forces are within actuator limits (0-250N from XML)
        left_force = np.clip(left_force, 0, 350)
        right_force = np.clip(right_force, 0, 350)
        
        # Set control signals for thruster actuators
        self.u.leftLeg.motorPd.torque[0] = left_force
        self.u.rightLeg.motorPd.torque[0] = right_force
        
        return left_force, right_force  # Return clipped values

    def run_visualization(self):
        """Run the visualization loop"""
        print("Starting real-time visualization...")
        print("Testing thruster forces (0-250N range)")
        
        t = 0
        last_update_time = time.time()
        frame_time = (1.0 / self.vis_freq) * self.slow_motion_factor

        while self.vis.valid():
            current_time = time.time()
            
            if current_time - last_update_time >= frame_time:
                for _ in range(self.steps_per_frame):
                    t += 1
                    self.step_counter += 1
                    
                    # Generate smoother oscillating forces
                    left_force = 175 + 175 * np.sin(t * self.dt * 0.5)  # Full range 0-250N
                    right_force = 175 + 175 * np.cos(t * self.dt * 0.5)
                    
                    # Apply forces through actuators
                    actual_left, actual_right = self.apply_thruster_forces(left_force, right_force)
                    
                    # Print forces every 100 steps
                    if self.step_counter % 100 == 0:
                        print(f"Left thruster: {actual_left:.2f}N, Right thruster: {actual_right:.2f}N")
                    
                    # Step simulation with PD controller
                    self.sim.step_pd(self.u)
                
                # Update visualization
                if not self.vis.draw(self.sim):
                    break
                
                last_update_time = current_time
            else:
                time.sleep(0.0001)

def main():
    visualizer = CassieVisualizer()
    visualizer.run_visualization()

if __name__ == "__main__":
    main()