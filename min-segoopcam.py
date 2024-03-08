import cv2
from ultralytics import YOLO
import numpy as np

class YOLOProcess:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.intrinsic = np.array([[1.4416e+03, 0, 581.3485], 
                                    [0, 1.4267e+03, 390.2253],
                                    [0, 0, 1]])
        self.extrinsic = np.array([[-0.0000000, -0.0000000, 1.0000000, 3],
                                   [0.8191521, 0.5735765, 0.0000000, 4],
                                   [-0.5735765, 0.8191521, -0.0000000, 5],
                                   [0, 0, 0, 1]])
        self.inverse_intrinsic = np.linalg.inv(self.intrinsic)
        self.inverse_extrinsic = np.linalg.inv(self.extrinsic)

    def process_camera(self, conf=0.25, imgsz=640):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(source=frame, conf=conf, imgsz=imgsz)
            
            for result in results:
                masks = result.masks
                for mask in masks:
                    segmentation_2d_point = np.array(mask.xy[0])
                    segmentation_hom_point_list = []

                    for i in range(segmentation_2d_point.shape[0]):
                        segmentation_hom_point = np.append(segmentation_2d_point[i], 1)
                        segmentation_hom_point_list.append(segmentation_hom_point)
                    
                    homogenous_point = np.array(segmentation_hom_point_list)
                    homogenous_point = homogenous_point.T

                    new_trans_point = self.inverse_intrinsic @ homogenous_point

                    ones_array = np.ones((segmentation_2d_point.shape[0],), dtype=np.float64)
                    new_trans_point = np.vstack((new_trans_point, ones_array))

                    world_point = self.inverse_extrinsic @ new_trans_point

                    drop_row = 3
                    pseudo_depth = np.delete(world_point, drop_row, axis=0).T

                    origin_point_of_camera = np.array([0, 0, 0, 1])
                    origin_in_real_world = self.inverse_extrinsic @ origin_point_of_camera

                    j = 2
                    coordinate_in_ground_plane = np.array([1, 1, 1])

                    for i in range(segmentation_2d_point.shape[0]):
                        alpha = (-4 - origin_in_real_world[j]) / pseudo_depth[i][j]
                        y_real = origin_in_real_world[j-1] + alpha * pseudo_depth[i][j-1]
                        x_real = origin_in_real_world[j-2] + alpha * pseudo_depth[i][j-2]
                        x_real = float(x_real) * 255
                        y_real = float(y_real) * 255
                        temporary_array = np.array([x_real, y_real, -4])
                        print(temporary_array)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo_process = YOLOProcess("segmentn.pt")
    yolo_process.process_camera(conf=0.25, imgsz=640)
