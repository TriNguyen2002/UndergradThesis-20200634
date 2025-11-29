#!/usr/bin/env python3
import rospy

from moveit_msgs.msg import RobotTrajectory
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from test_pack.custom_lib import CollisionMonitor
from ur5_interface.msg import HumanJoint
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray

USE_SIM = False
ERR_STD = np.array([[0.01 for i in range(6)]])
LOOP_MAX = 50
ANGLE_MAX = 0.025  # rad
DIST_rep = 0.05
DIST_att = 1.57
K_att = 1000
K_rep = 0.005

if USE_SIM:
    action_topic = "/ur5/eff_joint_traj_controller/follow_joint_trajectory"
    joint_topic = "/ur5/joint_states"
else:
    action_topic = "/scaled_pos_joint_traj_controller/follow_joint_trajectory"
    joint_topic = "/joint_states"


class ROSNode:
    def __init__(self):
        rospy.init_node("AAPF_NODE")
        rospy.loginfo("Starting AAPF_NODE.")
        self.planning_scene = CollisionMonitor()
        # previous pose for smoothing
        self.prev_pos = None
        # ensure action server is available before sending goals
        self.client = None
        self.setup_pub_sub()

    def setup_pub_sub(self):
        self.trajectory_sub = rospy.Subscriber("/ref_traj", RobotTrajectory, self.trajectory_callback)
        # action client is created here; wait for server availability
        # (moved creation to __init__ to allow waiting early)
        if self.client is None:
            self.client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
        # wait for up to 5 seconds, log warning if not available
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logwarn("FollowJointTrajectoryAction server not available after 5s. Commands may block.")

        self.visual_goal_pub = rospy.Publisher("/rviz_visual_tools", MarkerArray, queue_size=1)
        self.visual_goal_mod_pub = rospy.Publisher("/goal_mods", Marker, queue_size=1)

        self.replan_pub = rospy.Publisher("replan_RRT", HumanJoint, queue_size=1)

    def get_pos_curr(self) -> list:
        joint_msg = rospy.wait_for_message(joint_topic, JointState)
        joint_msg: JointState
        pos_curr = [
            joint_msg.position[2],
            joint_msg.position[1],
            joint_msg.position[0],
            joint_msg.position[3],
            joint_msg.position[4],
            joint_msg.position[5],
        ]
        return pos_curr

    def send_command(self, target_pos, vel):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names

        # build velocity array safely
        vel_list = [0, 0, 0, 0, 0, 0]
        try:
            if vel is not None:
                # vel can be a 1x6 array or 6-element array
                v = np.asarray(vel).reshape(-1)
                if v.size == 6 and np.all(np.isfinite(v)):
                    vel_list = v.tolist()
        except Exception:
            vel_list = [0, 0, 0, 0, 0, 0]

        goal.trajectory.points = [
            JointTrajectoryPoint(
                positions=target_pos.tolist()[0],
                velocities=vel_list,
                time_from_start=rospy.Duration(0.5),
            )
        ]

        # send trajectory and wait a short time for execution (non-blocking wait_for_result can stall main loop)
        try:
            self.client.send_goal(goal)
            # don't block too long; controller will execute the single-point trajectory
            self.client.wait_for_result(rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn(f"Failed sending trajectory goal: {e}")

    def visualize_ref_point(self, pos_ref_list):
        marker_array = MarkerArray()
        marker_clear = Marker()
        marker_clear.action = Marker.DELETEALL
        marker_array.markers.append(marker_clear)
        self.visual_goal_pub.publish(marker_array)
        marker_array.markers.clear()

        for idx, pos in enumerate(pos_ref_list):
            self.planning_scene.update_robot_goal(pos)

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = idx

            x, y, z = self.planning_scene.robot_goal_state[-1].getTranslation().tolist()
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z

            marker.scale.x = 0.075
            marker.scale.y = 0.075
            marker.scale.z = 0.075

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        self.visual_goal_pub.publish(marker_array)

    def visualize_goal_mod(self, pos_goal: list):
        self.planning_scene.update_robot_goal(pos_goal)
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = 20

        x, y, z = self.planning_scene.robot_goal_state[-1].getTranslation().tolist()
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        marker.scale.x = 0.075
        marker.scale.y = 0.075
        marker.scale.z = 0.075

        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.visual_goal_mod_pub.publish(marker)

    def trajectory_callback(self, robot_traj_msg):
        pos_ref_list = []
        self.joint_names = robot_traj_msg.joint_trajectory.joint_names
        for pt in robot_traj_msg.joint_trajectory.points:
            pt: JointTrajectoryPoint
            pos_ref_list.append(list(pt.positions))

        self.visualize_ref_point(pos_ref_list)
        # print(pos_ref_list)

        # Iterates through each ref points
        for pos_ref in pos_ref_list[1:]:
            pos_curr = np.array([self.get_pos_curr()])

            pos_goal = np.array([pos_ref])

            err_curr = np.abs(pos_curr - pos_goal)

            loop = 0

            while (np.any(err_curr > ERR_STD)) and loop < LOOP_MAX:
                #! Get Planning Scene
                human_msg = rospy.wait_for_message("human_position_topic", HumanJoint)
                self.planning_scene.update_planning_scene(pos_curr.tolist()[0], human_msg)

                #! Check pos_goal
                self.planning_scene.update_robot_goal(pos_goal.tolist()[0])
                self.planning_scene.compute_Jacob_goal(pos_goal.tolist()[0])
                goal_forces_list = []
                
                collision_flag = False
                for link in self.planning_scene.robot_goal_state:
                    if collision_flag:
                        break
                    rep_F = np.array([0.0, 0.0, 0.0])
                    for obs in self.planning_scene.static_env:
                        dist, rep_dir = self.planning_scene.compute_distance(obs, link)
                        if dist < 0:
                            collision_flag = True
                            print("COLLISION")
                            break
                        if dist < DIST_rep:
                            rep_F += K_rep * (1 / dist - 1 / DIST_rep) * 1 / (dist**2) * (rep_dir / dist)
                    goal_forces_list.append(np.transpose(rep_F))

                if collision_flag:
                    loop = loop + 1
                    continue

                goal_torques_rep = self.planning_scene.compute_torques_goal(goal_forces_list)
                if not all(element == 0.0 for element in goal_torques_rep):
                    pos_goal += 0.1 * (goal_torques_rep / np.linalg.norm(goal_torques_rep))
                    err_curr = np.abs(pos_curr - pos_goal)
                    continue
                self.visualize_goal_mod(pos_goal.tolist()[0])

                #! Compute torques
                self.planning_scene.compute_Jacob(pos_curr.tolist()[0])

                # Attractive
                torques_att = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                for i in range(6):
                    if err_curr[0, i] > ERR_STD[0, i]:
                            torques_att[0, i] = -K_att * (pos_curr[0, i] - pos_goal[0, i])
                
                # Repulsive
                forces_list = []
                for link in self.planning_scene.robot_curr_state:
                    rep_F = np.array([0.0, 0.0, 0.0])
                    for coll in self.planning_scene.static_env:
                        dist, rep_dir = self.planning_scene.compute_distance(coll, link)

                        if dist < DIST_rep:
                            rep_F += K_rep * (1 / dist - 1 / DIST_rep) * 1 / (dist**2) * (rep_dir / dist)

                    forces_list.append(np.transpose(rep_F))
                torques_rep = self.planning_scene.compute_torques(forces_list)
                torques_total = torques_att + torques_rep

                # Compute a stable step (delta) based on torque-derived suggestion but
                # scale proportional to magnitude and clip to a maximum step size.
                # This avoids always using a fixed step in the direction (causes jerks)
                torques_vec = torques_total.reshape(-1)
                norm_torques = np.linalg.norm(torques_vec)
                eps = 1e-6

                # scale factor to convert torque-like signal to angular step
                torque_to_step_gain = 0.02
                desired_step = torque_to_step_gain * torques_vec

                # clip total step magnitude to ANGLE_MAX (overall joint-space step)
                step_norm = np.linalg.norm(desired_step)
                if step_norm > ANGLE_MAX:
                    delta = desired_step / step_norm * ANGLE_MAX
                else:
                    delta = desired_step

                # per-joint clipping as additional safety
                delta = np.clip(delta, -ANGLE_MAX, ANGLE_MAX)

                # smoothing / low-pass filter between previous pos and new pos to reduce jitter
                if self.prev_pos is None:
                    self.prev_pos = pos_curr.copy()

                # candidate new position
                next_pos = (pos_curr.reshape(-1) + delta).reshape(1, -1)

                # smoothing factor alpha (0 < alpha <= 1). Lower alpha = smoother but slower.
                alpha = 0.6
                smoothed = alpha * next_pos + (1.0 - alpha) * self.prev_pos

                # update positions
                self.prev_pos = smoothed.copy()
                pos_curr = smoothed

                # compute commanded joint velocities from delta and chosen timeframe
                # ensure we avoid dividing by zero and send a sensible velocity vector to the controller
                timeframe = 0.5  # seconds used for each small incremental move (keeps motion slower)
                vel = (delta / max(timeframe, eps)).reshape(1, -1)

                #! Send Command to Robot (with velocity profile to reduce jerk)
                self.send_command(pos_curr, vel)

                #! Update err_curr
                pos_curr = np.array([self.get_pos_curr()])
                err_curr = np.abs(pos_curr - pos_goal)

                #! Increase Loop
                loop = loop + 1

            if np.any(err_curr > ERR_STD):
                print("RePlan")
                self.replan_pub.publish(human_msg)
                break


if __name__ == "__main__":
    name_node = ROSNode()
    rospy.spin()
