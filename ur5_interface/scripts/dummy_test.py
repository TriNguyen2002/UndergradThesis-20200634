#!/usr/bin/env python3
#!/usr/bin/env python3
import rospy
import random
import math
from ur5_interface.msg import HumanJoint
from visualization_msgs.msg import Marker

if __name__ == "__main__":
    rospy.init_node("Dummy_node")
    rospy.loginfo("Starting Dummy_node.")

    dummy_pub = rospy.Publisher("human_position_topic", HumanJoint, queue_size=1)
    marker_dummy = rospy.Publisher("/human_topic_marker", Marker, queue_size=1)

    dummy_msg = HumanJoint()
    dummy_msg.exist = True
    
    # --- Replan in plan_3.pickle ---
    # dummy_msg.position.x = 0.5
    # dummy_msg.position.y = 0.45
    # dummy_msg.position.z = 0.3
    # ----------------------------------

    # --- plan_tri_1.pickle ---
    dummy_msg.position.x = 0.55 # o.5 for replan case
    dummy_msg.position.y = -0.4
    dummy_msg.position.z = 0.24
    # ----------------------------------

    dummy_marker = Marker()
    dummy_marker.header.frame_id = "base_link"
    dummy_marker.type = Marker.SPHERE
    dummy_marker.action = Marker.ADD
    dummy_marker.id = 1
    dummy_marker.pose.position.x = dummy_msg.position.x
    dummy_marker.pose.position.y = dummy_msg.position.y
    dummy_marker.pose.position.z = dummy_msg.position.z
    dummy_marker.pose.orientation.w = 1
    dummy_marker.scale.x = 0.109
    dummy_marker.scale.y = 0.109
    dummy_marker.scale.z = 0.109
    dummy_marker.color.r = 0.0
    dummy_marker.color.g = 0.0
    dummy_marker.color.b = 1.0
    dummy_marker.color.a = 0.5
    dummy_marker.lifetime = rospy.Duration(2)

    # configuration (override with ROS params if desired)
    jitter_amplitude = rospy.get_param("~jitter_amplitude", 0.01)  # meters (stddev if random)
    jitter_frequency = rospy.get_param("~jitter_frequency", 0.8)  # Hz for sine jitter
    jitter_method = rospy.get_param("~jitter_method", "random")  # 'random' or 'sine'
    publish_rate_hz = rospy.get_param("~publish_rate_hz", 10)

    rate = rospy.Rate(publish_rate_hz)

    # baseline (original) positions — keep these and add jitter on top
    base_x, base_y, base_z = (dummy_msg.position.x, dummy_msg.position.y, dummy_msg.position.z)

    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        if dummy_msg.exist:
            if jitter_method == "sine":
                t = rospy.get_time() - start_time
                # create small sine waves with phase offsets per axis for natural motion
                dx = jitter_amplitude * math.sin(2 * math.pi * jitter_frequency * t)
                dy = jitter_amplitude * math.sin(2 * math.pi * jitter_frequency * t + math.pi / 3)
                dz = jitter_amplitude * math.sin(2 * math.pi * jitter_frequency * t + math.pi / 6)
            else:
                # random jitter around baseline (Gaussian noise)
                dx = random.gauss(0.0, jitter_amplitude)
                dy = random.gauss(0.0, jitter_amplitude)
                dz = random.gauss(0.0, jitter_amplitude)

            # update message and marker
            dummy_msg.position.x = base_x + dx
            dummy_msg.position.y = base_y + dy
            dummy_msg.position.z = base_z + dz

            dummy_marker.pose.position.x = dummy_msg.position.x
            dummy_marker.pose.position.y = dummy_msg.position.y
            dummy_marker.pose.position.z = dummy_msg.position.z

        # publish
        dummy_pub.publish(dummy_msg)
        if dummy_msg.exist:
            marker_dummy.publish(dummy_marker)

        rate.sleep()
