#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import os
import rospkg
import numpy as np
from collections import deque
import math
from visualization_msgs.msg import Marker, MarkerArray
from ur5_interface.msg import HumanJoint
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf.transformations


class ROSNode:
    def __init__(self):
        rospy.init_node("YOLO_NODE")
        rospy.loginfo("Starting YOLO_NODE.")
        self.setup_model()
        self.setup_pub_sub()

    def setup_model(self):
        # YOLO_MODEL
        model_path = os.path.join(rospkg.RosPack().get_path("ur5_interface"), "models/yolov8s-pose.pt")
        self.model = YOLO(model_path)
        self.model.fuse()
        # Camera Info
        camera_info = rospy.wait_for_message("/camera/aligned_depth_to_color/camera_info", CameraInfo)
        camera_info: CameraInfo
        self.camera_matrix = np.array(camera_info.K, dtype=np.float32).reshape(3, 3)
        print(self.camera_matrix)

        # Bridge
        self.bridge = CvBridge()

        # smoothing / filtering params
        # small history + exponential smoothing (tunable)
        self.smoothing_alpha = rospy.get_param("~smoothing_alpha", 0.6)
        self.history_size = rospy.get_param("~history_size", 5)
        # maximum allowed jump (meters) between frames - large jumps will be clamped
        self.max_jump_m = rospy.get_param("~max_jump_m", 0.5)

        # history of positions (in base_link frame) for smoothing & outlier handling
        self.position_history = deque(maxlen=self.history_size)

        # defaults for neighborhood depth median
        self.depth_kernel = rospy.get_param("~depth_kernel", 3)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def setup_pub_sub(self):
        self.img_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_callback)
        self.img_pub_debug = rospy.Publisher("/img_debug", Image, queue_size=2)
        self.joint_pub = rospy.Publisher("/human_position_topic", HumanJoint, queue_size=10)
        self.marker_human = rospy.Publisher("/human_topic_marker", Marker, queue_size=1)

    def _median_depth_at(self, depth_array: np.ndarray, x_pixel: float, y_pixel: float, kernel: int = None):
        """Return median depth in meters for a small neighborhood centered at pixel.

        Returns None when no valid depth values are present in the neighborhood.
        """
        if kernel is None:
            kernel = int(self.depth_kernel)
        k = max(1, int(kernel))
        half = k // 2

        h, w = depth_array.shape
        x = int(round(x_pixel))
        y = int(round(y_pixel))

        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)

        patch = depth_array[y0:y1, x0:x1].flatten()
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0]
        if patch.size == 0:
            return None
        # depth values are mm in the camera topic; convert to meters
        return float(np.median(patch)) / 1000.0

    def _smooth_and_clamp(self, new_pos: np.ndarray):
        """Clamp large jumps and apply exponential smoothing.

        new_pos: numpy array shape (3,) in meters
        Returns smoothed position (3,) and stores latest value in history
        """
        if not isinstance(new_pos, np.ndarray):
            new_pos = np.asarray(new_pos, dtype=float)

        if len(self.position_history) == 0:
            self.position_history.append(new_pos.copy())
            return new_pos.copy()

        prev = self.position_history[-1]
        diff = new_pos - prev
        dist = np.linalg.norm(diff)
        if dist > self.max_jump_m and dist > 1e-6:
            # clamp the change to max_jump_m to avoid teleports
            diff = diff * (self.max_jump_m / dist)
            new_pos = prev + diff

        alpha = float(self.smoothing_alpha)
        smoothed = alpha * new_pos + (1.0 - alpha) * prev
        self.position_history.append(smoothed.copy())
        return smoothed

    def img_callback(self, img_msg: Image):
        img_cv = self.bridge.imgmsg_to_cv2(img_msg)
        img_depth = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)

        results = self.model.track(
            source=img_cv,
            conf=0.7,
            iou=0.8,
            max_det=10,
            device="cpu",
            classes=0,
            stream=False,
            verbose=False,
        )

        plotted_image = results[0].plot(
            conf=True,
            line_width=1,
            font_size=1,
            font="Arial.ttf",
            labels=True,
            boxes=True,
        )

        if len(results[0].boxes):
            conf = results[0].keypoints.cpu().numpy().conf.tolist()[0]
            points = results[0].keypoints.cpu().numpy().xy.tolist()[0]
            human_joint_list = list(zip(conf, points))
            if human_joint_list[10][0] >= 0.8 or human_joint_list[9][0] >= 0.8:
                self.compute_joint_3D(human_joint_list, img_depth)
        else:
            human_msg = HumanJoint()
            human_msg.exist = False
            self.joint_pub.publish(human_msg)

        self.send_img_debug(plotted_image)

    def compute_joint_3D(self, human_joint_list, img_depth):
        # Depth denoising: median in a small neighborhood
        img_depth_cv = self.bridge.imgmsg_to_cv2(img_depth, desired_encoding="passthrough")
        depth_array = np.array(img_depth_cv, dtype=np.float32)

        # prefer index 10 (shoulder/right) falling back to 9
        idx = 10 if human_joint_list[10][0] >= 0.0 else 9
        x_pixel = human_joint_list[idx][1][0]
        y_pixel = human_joint_list[idx][1][1]

        median_depth = self._median_depth_at(depth_array, x_pixel, y_pixel)
        if median_depth is None:
            # no valid depth in neighborhood; publish nobody
            human_msg = HumanJoint()
            human_msg.exist = False
            self.joint_pub.publish(human_msg)
            return

        # compute position in camera coordinates
        joint_depth = float(median_depth)
        x_real = float((x_pixel - self.camera_matrix[0, 2]) / float(self.camera_matrix[0, 0])) * joint_depth
        y_real = float((y_pixel - self.camera_matrix[1, 2]) / float(self.camera_matrix[1, 1])) * joint_depth

        human_mtx = np.asarray([[x_real, y_real, joint_depth, 1.0]]).reshape(4, 1)

        # Transform to base_link (use try/except in case TF not ready)
        try:
            trans = self.tfBuffer.lookup_transform("base_link", "camera_link", rospy.Time(), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn_once(f"TF lookup failed for camera->base_link: {e}")
            # If transform missing, publish raw camera-frame results with exist True
            human_msg = HumanJoint()
            human_msg.position.x = x_real
            human_msg.position.y = y_real
            human_msg.position.z = joint_depth
            human_msg.exist = True
            self.joint_pub.publish(human_msg)

            # also publish a camera-frame marker for debugging
            human_marker = Marker()
            human_marker.header.frame_id = "camera_link"
            human_marker.type = Marker.SPHERE
            human_marker.action = Marker.ADD
            human_marker.id = 12
            human_marker.pose.position.x = x_real
            human_marker.pose.position.y = y_real
            human_marker.pose.position.z = joint_depth
            human_marker.pose.orientation.w = 1
            human_marker.scale.x = 0.218
            human_marker.scale.y = 0.218
            human_marker.scale.z = 0.218
            human_marker.color.r = 1.0
            human_marker.color.a = 1.0
            human_marker.lifetime = rospy.Duration(2)
            self.marker_human.publish(human_marker)
            return

        trans: TransformStamped
        color_from_base_link = tf.transformations.quaternion_matrix(
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ]
        )
        color_from_base_link[0, 3] = trans.transform.translation.x
        color_from_base_link[1, 3] = trans.transform.translation.y
        color_from_base_link[2, 3] = trans.transform.translation.z

        human_from_base_link = np.dot(color_from_base_link, human_mtx).reshape(4,)

        new_pos = np.array([float(human_from_base_link[0]), float(human_from_base_link[1]), float(human_from_base_link[2])])

        # smoothing + clamp outliers
        smoothed = self._smooth_and_clamp(new_pos)

        human_msg = HumanJoint()
        human_msg.position.x = float(smoothed[0])
        human_msg.position.y = float(smoothed[1])
        human_msg.position.z = float(smoothed[2])
        human_msg.exist = True

        self.joint_pub.publish(human_msg)

        # publish smoothed marker in base_link frame (reduces visual jitter)
        human_marker = Marker()
        human_marker.header.frame_id = "base_link"
        human_marker.type = Marker.SPHERE
        human_marker.action = Marker.ADD
        human_marker.id = 12
        human_marker.pose.position.x = smoothed[0]
        human_marker.pose.position.y = smoothed[1]
        human_marker.pose.position.z = smoothed[2]
        human_marker.pose.orientation.w = 1
        human_marker.scale.x = 0.218
        human_marker.scale.y = 0.218
        human_marker.scale.z = 0.218
        human_marker.color.r = 1.0
        human_marker.color.g = 0.0
        human_marker.color.b = 0.0
        human_marker.color.a = 1.0
        human_marker.lifetime = rospy.Duration(2)

        self.marker_human.publish(human_marker)

    def send_img_debug(self, img_cv):
        img_msg = self.bridge.cv2_to_imgmsg(img_cv, encoding="rgb8")
        self.img_pub_debug.publish(img_msg)


if __name__ == "__main__":
    name_node = ROSNode()
    rospy.spin()
