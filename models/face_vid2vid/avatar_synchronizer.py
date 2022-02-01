import torch
import numpy as np
from models.face_vid2vid.utils import synchronize_avatar


class AvatarSynchronizer:
    def __init__(self) -> None:
        super(AvatarSynchronizer, self).__init__()
        self.activated = False
        self.driving = None
        self.kp_canonical = None
        self.kp_source = None
        self.kp_driving_initial = None

    def activate(self, avatar_tensor, frame, kp_detector, he_estimator):
        self.driving = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        self.kp_canonical, self.kp_source, self.kp_driving_initial = \
            synchronize_avatar(self.driving, kp_detector, he_estimator, avatar_tensor)
        self.activated = True
