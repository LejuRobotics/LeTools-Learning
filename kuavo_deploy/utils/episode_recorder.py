"""DAgger bag recording and one compact marker per successful command."""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

DECISION_TOPIC = "/dagger/decision"


class DecisionRecorder:
    def __init__(self):
        self.publisher = None
        self.episode_id = os.environ.get("DAGGER_EPISODE_ID")
        if self.episode_id:
            import rospy
            from std_msgs.msg import String
            # rospy's bounded asynchronous publisher keeps socket writes off the control loop.
            self.publisher = rospy.Publisher(DECISION_TOPIC, String, queue_size=1000)
            deadline = time.monotonic() + 3.0
            while self.publisher.get_num_connections() == 0:
                if rospy.is_shutdown() or time.monotonic() >= deadline:
                    self.close()
                    raise RuntimeError("DAgger decision recorder is not connected")
                time.sleep(0.02)

    def record(self, source):
        if self.publisher is None:
            return
        import rospy
        from std_msgs.msg import String
        parts = source.split("+")
        event = dict(schema_version=4, episode_id=self.episode_id,
                     stamp_ns=rospy.Time.now().to_nsec(), source=source)
        for side in ("left", "right", "waist"):
            event[side + "_intervention"] = any(p in (side + "_blend", side + "_vr") for p in parts)
        self.publisher.publish(String(data=json.dumps(event, allow_nan=False)))

    def close(self):
        if self.publisher is not None:
            self.publisher.unregister()
            self.publisher = None


def recording_topics(config):
    env, inf = config.env, config.inference
    observations = getattr(env, "recording_obs_key_map", None) or env.obs_key_map
    topics = [spec["topic"] for spec in observations.values() if "topic" in spec]
    for spec in observations.values():
        if spec.get("msg_type", "").rsplit("/", 1)[-1] in ("Image", "CompressedImage"):
            topic = spec["topic"]
            for suffix in ("/compressed", "/compressedDepth", "/theora"):
                if topic.endswith(suffix):
                    topic = topic[:-len(suffix)]
            topics.append(topic.rsplit("/", 1)[0] + "/camera_info")
    topics.extend(("/kuavo_arm_traj", "/joint_cmd", "/kuavo/pause_state", "/kuavo/stop_state",
                   inf.dagger_vr_candidate_topic, inf.dagger_vr_eef_topic, DECISION_TOPIC))
    sides = ("left", "right") if env.which_arm == "both" else (env.which_arm,)
    topics.extend(getattr(inf, f"dagger_{side}_intervention_topic") for side in sides)
    topics.append({"qiangnao": "/control_robot_hand_position", "leju_claw": "/leju_claw_command",
                   "rq2f85": "/gripper/command", "sg100": "/sg100_hand_command"}[env.eef_type])
    if env.use_5w_wholebody:
        topics.append("/lb_leg_traj")
    if env.use_5w_base_move:
        topics.append("/cmd_vel")
    if inf.dagger_waist_on:
        topics.append("/dagger/vr/waist_config")
    return sorted(set(topics + inf.rosbag_extra_topics))


class EpisodeRecorder:
    def __init__(self, config, repo_root):
        self.config = config
        self.repo_root = Path(repo_root)
        self.proc = None
        self.log = None
        self.metadata = None
        self.closed = False
        self.last_health_check = 0.0

    def _write_metadata(self):
        path = self.directory / "metadata.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temp.replace(path)

    def start(self):
        import rospy
        import roslib.packages
        from dataclasses import asdict
        inf = self.config.inference
        root = Path(inf.rosbag_output_dir).expanduser()
        if not root.is_absolute():
            root = self.repo_root / root
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < inf.rosbag_min_free_gb * 1024**3:
            raise RuntimeError(f"录包目录不足 {inf.rosbag_min_free_gb} GiB 可用空间: {root}")
        now = datetime.datetime.now().astimezone()
        episode_id = now.strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
        self.directory = root / now.strftime("%Y%m%d") / episode_id
        self.directory.mkdir(parents=True, exist_ok=False)
        # Keep date/*.bag compatible with the unchanged dev converter.
        self.bag_path = self.directory.with_suffix(".bag")
        topics = recording_topics(self.config)
        self.metadata = dict(schema_version=4, episode_id=episode_id, created_at=now.isoformat(),
                             bag_path=str(self.bag_path), config=asdict(self.config), topics=topics,
                             bag_closed=False)
        try:
            self._write_metadata()
            if not rospy.core.is_initialized():
                rospy.init_node("dagger_recording_launcher", anonymous=True, disable_signals=True)
            if rospy.Time.now().to_nsec() == 0:
                raise RuntimeError("ROS time is zero; refusing an untimestamped recording")
            nodes = roslib.packages.find_node("rosbag", "record")
            if not nodes:
                raise RuntimeError("Cannot locate ROS1 rosbag record executable")
            name = "dagger_bag_" + uuid.uuid4().hex
            args = [nodes[0], "--lz4", "--buffsize", str(inf.rosbag_buffer_mb),
                    "--min-space", f"{int(inf.rosbag_min_free_gb * 1024)}M",
                    "-O", str(self.bag_path), "__name:=" + name, *topics]
            self.log = (self.directory / "recorder.log").open("w")
            self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                self.check_health()
                _, subscriptions, _ = rospy.get_master().getSystemState()[2]
                ready = {topic for topic, names in subscriptions if "/" + name in names}
                if set(topics) <= ready and Path(str(self.bag_path) + ".active").exists():
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("rosbag recorder subscription startup timed out")
            self.metadata["start_ros_ns"] = rospy.Time.now().to_nsec()
            self._write_metadata()
            print(f"🔴 录制到: {self.bag_path}", flush=True)
            return episode_id
        except BaseException:
            try:
                self.finish("startup_error", None)
            except Exception as exc:
                print(f"录包启动失败后的清理: {exc}", flush=True)
            raise

    def check_health(self):
        if self.proc is not None and self.proc.poll() is not None:
            raise RuntimeError(f"rosbag recorder exited: {self.proc.returncode}")
        now = time.monotonic()
        if now - self.last_health_check >= 1.0:
            self.last_health_check = now
            if shutil.disk_usage(self.directory).free < self.config.inference.rosbag_min_free_gb * 1024**3:
                raise RuntimeError("录包磁盘空间不足，停止本次采集")

    def finish(self, reason, task_returncode):
        if self.closed:
            return
        self.closed = True
        error = None
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                    error = "rosbag close timed out"
        finally:
            if self.log is not None:
                self.log.close()
            if self.metadata is not None:
                import rospy
                closed = self.bag_path.exists() and not Path(str(self.bag_path) + ".active").exists()
                recorder_code = self.proc.returncode if self.proc else None
                self.metadata.update(end_ros_ns=rospy.Time.now().to_nsec(), stop_reason=reason,
                                     task_returncode=task_returncode, recorder_returncode=recorder_code,
                                     bag_closed=closed)
                if not closed or recorder_code not in (None, 0):
                    error = error or f"rosbag 未正常结束: exit={recorder_code}, bag_closed={closed}"
                if error:
                    self.metadata["error"] = error
                self._write_metadata()
        if error:
            raise RuntimeError(error)
