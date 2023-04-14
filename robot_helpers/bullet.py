import numpy as np
import pybullet as p

from .perception import CameraIntrinsic
from .spatial import Rotation, Transform

assert p.isNumpyEnabled(), "Pybullet needs to be built with NumPy"


class BtPandaArm:
    def __init__(self, urdf_path="franka_panda/panda.urdf", pose=Transform.identity()):
        self.base_frame = "panda_link0"
        self.ee_frame = "panda_hand"
        #self.configurations = {"ready": [0.0, -0.79, 0.0, -2.356, 0.0, 1.57, 0.79]}
        self.configurations = {"ready": [0.0, -1.3, 0.0, -2.2, 0.0, 1.4, 0.79+(np.pi/2)]}
        self.uid = p.loadURDF(
            str(urdf_path),
            basePosition=pose.translation,
            baseOrientation=pose.rotation.as_quat(),
            useFixedBase=True,
        )
        for i, q_i in enumerate(self.configurations["ready"]):
            p.resetJointState(self.uid, i, q_i)

    def get_state(self):
        joint_states = p.getJointStates(self.uid, range(p.getNumJoints(self.uid)))[:7]
        q = np.asarray([state[0] for state in joint_states])
        dq = np.asarray([state[1] for state in joint_states])
        return q, dq

    def set_desired_joint_positions(self, q):
        for i, q_i in enumerate(q):
            p.setJointMotorControl2(self.uid, i, p.POSITION_CONTROL, q_i)

    def set_desired_joint_velocities(self, dq):
        for i, dq_i in enumerate(dq):
            p.setJointMotorControl2(
                self.uid, i, p.VELOCITY_CONTROL, targetVelocity=dq_i
            )


class BtPandaGripper:
    def __init__(self, arm):
        self.uid = arm.uid
        # Constraint to keep the fingers centered
        uid = p.createConstraint(
            self.uid,
            9,
            self.uid,
            10,
            p.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        p.changeConstraint(uid, gearRatio=-1, erp=0.1, maxForce=50)

    def set_desired_width(self, width, force=5):
        p.setJointMotorControlArray(
            self.uid,
            [9, 10],
            p.POSITION_CONTROL,
            [0.5 * width] * 2,
            forces=[force] * 2,
        )

    def set_desired_speed(self, speed, force=5):
        p.setJointMotorControlArray(
            self.uid,
            [9, 10],
            p.VELOCITY_CONTROL,
            targetVelocities=[speed] * 2,
            forces=[force] * 2,
        )

    def read(self):
        left_pos = p.getJointState(self.uid, 9)[0]
        right_pos = p.getJointState(self.uid, 10)[0]
        return left_pos + right_pos


class BtCamera:
    def __init__(
        self,
        width,
        height,
        vfov,
        near,
        far,
        body_uid=None,
        link_id=None,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        rot = 1
    ):
        f, cx, cy = height / (2.0 * np.tan(vfov / 2.0)), width / 2.0, height / 2.0
        self.intrinsic = CameraIntrinsic(width, height, f, f, cx, cy)
        self.near = near
        self.far = far
        fov, aspect = np.rad2deg(vfov), width / height
        self.proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        self.body_uid = body_uid
        self.link_id = link_id
        self.renderer = renderer
        self.rot = rot

    def get_image(self, pose=None):
        if pose is None:
            r = p.getLinkState(self.body_uid, self.link_id, computeForwardKinematics=1)
            pose = Transform(Rotation.from_quat(r[5]), r[4])
            theta = self.rot
            rotation_matrix = np.array(
            # # [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]#chatgpt Y rotation
            # #[[0, 0, -1], [0, 1, 0], [0, 0, -1]]#Ry -90 deg 
            # # [[0, -1, 0], [1, 0, 0], [0, 0, 1]]#chatgpt Z rotation
            # [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]#chatgpt X rotation
            # [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta),0, np.cos(theta)]] # Y
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]] # Z
            )
            # print(rotation_matrix)

            pose.rotation = Rotation.from_matrix(rotation_matrix) * pose.rotation

 

        # rot, trans = pose.rotation, pose.translation

        # print("rot", rot)
        # #pose.rotation = rot.apply([1, 0, 0])
        # pose.apply([1,0,0])
        R, t = pose.rotation, pose.translation
        #print(R)
        #print(self.rot)

        # if self.rot%2 != 0:
        #view_mat = p.computeViewMatrix(t, R.apply([0, 0, 1]) + t, R.apply([1, 0, 0]))
        # else:
        view_mat = p.computeViewMatrix(t, R.apply([0, 0, 1]) + t, R.apply([0, -1, 0])) #original 

        # pose = Transform.from_matrix(pose.as_matrix() * np.reshape(view_mat, (4,4)))
        self.pose = pose 
        #view_mat = p.computeViewMatrixFromYawPitchRoll()

        result = p.getCameraImage(
            self.intrinsic.width,
            self.intrinsic.height,
            view_mat,
            self.proj_mat,
            renderer=self.renderer,
        )
        color = result[2][:, :, :3]
        depth = self.far * self.near / (self.far - (self.far - self.near) * result[3])
        mask = result[4]
        return color, depth, mask
