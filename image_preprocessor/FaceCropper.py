import cv2
import numpy as np
import mediapipe as mp

class FaceCropper:
    def __init__(self) -> None:
        self.model_name: str = 'Face Detection through mediapipe'
        self.model_version: str = '1.0.0'
        self.model_description: str = 'Face Detection through mediapipe'

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.model_status = bool(True)
        self.landmarks = []

        self.FACE_CENTER = 4
        self.FACE_TOP = 10
        self.FACE_BOTTOM = 377
        self.FACE_LEFT = 123
        self.FACE_RIGHT = 352
        self.CUT_EXTRA = 50

        self.previous_nose_point = [0, 0]
        self.cut_points = {'top_point': [0, 0], 'bottom_point': [0, 0], 'left_point': [0, 0], 'right_point': [0, 0]}
        self.previous_cut_points = {'top_point': [0, 0], 'bottom_point': [0, 0], 'left_point': [0, 0], 'right_point': [0, 0]}

    def get_facial_landmarks(self, input_frame: np.ndarray) -> np.ndarray:
        input_frame: np.ndarray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        inference_result: object = self.face_mesh.process(input_frame)

        if inference_result.multi_face_landmarks:
            landmarks_list = list()
            for facial_landmarks in inference_result.multi_face_landmarks:
                for index in range(0, 468):
                    landmark_point: mp.framework = facial_landmarks.landmark[index]
                    x, y = int(landmark_point.x * input_frame.shape[1]), int(landmark_point.y * input_frame.shape[0])
                    landmarks_list.append([x, y])

            del inference_result  # memory free
            self.landmarks = np.array(landmarks_list, np.int32)
            return self.landmarks
        else:
            return None

    ''' Returns Image Cut Points based on Face Location '''
    def get_cut_point(self) -> dict:
        top_point = self.landmarks[self.FACE_TOP][1]-self.CUT_EXTRA
        bottom_point = self.landmarks[self.FACE_BOTTOM][1]+self.CUT_EXTRA
        left_point = self.landmarks[self.FACE_LEFT][0]-self.CUT_EXTRA
        right_point = self.landmarks[self.FACE_RIGHT][0]+self.CUT_EXTRA

        is_point_out_of_image = list(map(lambda x: True if x <= 0 else False, [top_point, bottom_point, left_point, right_point]))

        # if index is out of range, return previous points
        if any(is_point_out_of_image):
            return self.previous_cut_points

        self.previous_cut_points = {'top_point': top_point, 'bottom_point': bottom_point, 'left_point': left_point, 'right_point': right_point}
        return {'top_point': top_point, 'bottom_point': bottom_point, 'left_point': left_point, 'right_point': right_point}

    ''' Start Cropping Image '''
    def crop_image(self, input_frame: np.ndarray) -> np.ndarray:
        if self.nose_moved_distance() > 3: # Prevents Frame Flickering
            self.cut_points = self.get_cut_point()

        cropped_frame = input_frame[self.cut_points["top_point"]:self.cut_points["bottom_point"], self.cut_points["left_point"]:self.cut_points["right_point"]]
        self.previous_nose_point = self.get_nose_coordinate()

        return cropped_frame

    ''' Calculate Nose Distance Moved '''
    def nose_moved_distance(self) -> int:
        return abs(self.previous_nose_point[0] - self.landmarks[self.FACE_CENTER][0]) + abs(self.previous_nose_point[1] - self.landmarks[self.FACE_CENTER][1])

    ''' Get Nose Coordinate '''
    def get_nose_coordinate(self) -> np.ndarray:
        return self.landmarks[self.FACE_CENTER]
