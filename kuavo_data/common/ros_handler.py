# Copyright (C) 2025-2026 LejuRobotics.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This project includes code from LeRobot (https://github.com/huggingface/lerobot),
# which is licensed under the Apache License, Version 2.0.

import rospy

from multiprocessing import Process, Queue
import time
import json

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CompressedImage
from kuavo_msgs.msg import sensorsData, jointCmd
from rospy_message_converter import message_converter



class ROSHandler:
    def __init__(self, queue):
        self.queue = queue
        self.topics_config = [
            ('/sensors_data_raw', sensorsData),   
            ('/joint_cmd', jointCmd),            
            ('/cam_h/color/compressed', CompressedImage),
            ('/cam_l/color/compressed', CompressedImage),
            ('/cam_r/color/compressed', CompressedImage)
        ]

    def _extract_data(self, msg, topic_name):
        """根据话题类型提取关键数据"""
        data = {}
        try:
            msg_dict = message_converter.convert_ros_message_to_dictionary(msg)
            if topic_name == '/sensors_data_raw':
                data = msg_dict
            elif topic_name == '/joint_cmd':
                data = msg_dict
            elif topic_name ==  '/cam_h/color/compressed':
                data = msg_dict
            elif topic_name == '/cam_l/color/compressed':
                data = msg_dict
            elif topic_name == '/cam_r/color/compressed':
                data = msg_dict
                # rospy.loginfo(f"Received image message: {data.keys()}")
        except AttributeError as e:
            rospy.logwarn(f"Missing expected field in {topic_name} message: {str(e)}")
        return data

    def _get_timestamp(self, msg):
        """从消息头获取时间戳，回退到当前时间"""
        try:
            return msg.header.stamp.to_sec()
        except AttributeError:
            return rospy.get_time()

    def _generic_callback(self, msg, topic_name):
        """统一消息回调处理"""
        try:
            message = {
                'type': topic_name,
                'data': self._extract_data(msg, topic_name),
                'timestamp': self._get_timestamp(msg)
            }
            self.queue.put(json.dumps(message))
            # rospy.loginfo(f"Message processed for {topic_name}")
        except Exception as e:
            rospy.logerr(f"Message processing failed for {topic_name}: {str(e)}")

    def run(self):
        """启动ROS节点并订阅所有配置的话题"""
        rospy.init_node('ros_handler', anonymous=True)
        
        # 动态创建消息类型映射
        msg_type_mapping = {topic: msg_type for topic, msg_type in self.topics_config}
        
        for topic_name, _ in self.topics_config:
            # 使用闭包正确捕获当前topic_name的值
            rospy.Subscriber(
                topic_name,
                msg_type_mapping[topic_name],
                lambda msg, tn=topic_name: self._generic_callback(msg, tn)
            )
        rospy.spin()

def start_ros_handler(queue):
    handler = ROSHandler(queue)
    handler.run()


def get_ros_queue(maxsize=10):
    """初始化ROS消息队列"""
    queue = Queue(maxsize=maxsize)
    p = Process(target=start_ros_handler, args=(queue,))
    p.start()
    return queue, p

if __name__ == "__main__":
    queue, p = get_ros_queue()
    
    try:
        while True:
            time.sleep(1)  # Keep main process alive
            print(queue.qsize())
    except KeyboardInterrupt:
        p.terminate()
        p.join()