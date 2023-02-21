import supervisely as sly
from supervisely.geometry.graph import KeypointsTemplate
from typing_extensions import Literal
from typing import List, Any, Dict, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import torch
import torchvision
from dotenv import load_dotenv
import cv2
from mmpose.apis import inference_top_down_pose_model, init_pose_model
import numpy as np
import os


root_source_path = str(Path(__file__).parents[2])
app_source_path = str(Path(__file__).parents[1])
model_data_path = os.path.join(root_source_path, "models", "model_data.json")
table_data_path = os.path.join(root_source_path, "models", "table_data.json")
load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


class ViTPoseModel(sly.nn.inference.PoseEstimation):
    def get_models(self):
        table_data = sly.json.load_json_file(table_data_path)
        return table_data

    def get_models_data(self):
        model_data = sly.json.load_json_file(model_data_path)
        models_data_processed = {}
        for element in model_data:
            config_path = os.path.join(root_source_path, "configs", element["config_file_name"])
            models_data_processed[element["model_name"]] = {
                "config": config_path,
                "weights": element["weights_link"],
            }
        return models_data_processed

    def get_weights_and_config_path(self, model_dir):
        model_source = self.gui.get_model_source()
        weights_dst_path = os.path.join(model_dir, "weights.pth")
        if model_source == "Pretrained models":
            models_data = self.get_models_data()
            selected_model = self.gui.get_model_info()[0]
            weights_link = models_data[selected_model]["weights"]
            if not sly.fs.file_exists(weights_dst_path):
                self.download(src_path=weights_link, dst_path=weights_dst_path)
            config_path = models_data[selected_model]["config"]
            return weights_dst_path, config_path
        elif model_source == "Custom weights":
            custom_link = self.gui.get_custom_link()
            if not sly.fs.file_exists(weights_dst_path):
                self.download(
                    src_path=custom_link,
                    dst_path=weights_dst_path,
                )
            config_path = self.download(
                src_path=os.path.join(os.path.dirname(custom_link), "pose_config.py"),
                dst_path=os.path.join(model_dir, "pose_config.py"),
            )
            return weights_dst_path, config_path

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # define model config and checkpoint
        if sly.is_production():
            pose_checkpoint, pose_config = self.get_weights_and_config_path(model_dir)
        else:
            # for local debug only
            models_data = self.get_models_data()
            weights_link = models_data[selected_model]["weights"]
            pose_checkpoint = os.path.join(model_dir, "weights.pth")
            if not sly.fs.file_exists(pose_checkpoint):
                sly.fs.download(url=weights_link, save_path=pose_checkpoint)
            pose_config = models_data[selected_model]["config"]
        # initialize pose estimator
        self.pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
        # define class names
        self.class_names = ["person_keypoints"]
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self):
        return self.class_names

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        return info

    def predict(
        self, image_path: str, settings: Dict[str, Any]
    ) -> List[sly.nn.PredictionKeypoints]:
        # transfer crop from annotation tool to bounding box
        input_image = sly.image.read(image_path)
        img_height, img_width = input_image.shape[:2]
        bbox = [{"bbox": np.array([0, 0, img_width, img_height, 1.0])}]

        # get point labels
        point_labels = self.keypoints_template.point_names

        # inference pose estimator
        if "local_bboxes" in settings:
            bboxes = settings["local_bboxes"]
        elif "detected_bboxes" in settings:
            bboxes = settings["detected_bboxes"]
            for i in range(len(bboxes)):
                box = bboxes[i]["bbox"]
                bboxes[i] = {"bbox": np.array(box)}
        else:
            bboxes = bbox

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            image_path,
            bboxes,
            format="xyxy",
            dataset=self.pose_model.cfg.data.test.type,
        )

        # postprocess results
        point_threshold = settings.get("point_threshold", 0.01)
        results = []
        for result in pose_results:
            included_labels, included_point_coordinates = [], []
            point_coordinates, point_scores = result["keypoints"][:, :2], result["keypoints"][:, 2]
            for i, (point_coordinate, point_score) in enumerate(
                zip(point_coordinates, point_scores)
            ):
                if point_score >= point_threshold:
                    included_labels.append(point_labels[i])
                    included_point_coordinates.append(point_coordinate)
            results.append(
                sly.nn.PredictionKeypoints(
                    "person_keypoints", included_labels, included_point_coordinates
                )
            )
        return results


# build keypoints template
# initialize template
template = KeypointsTemplate()
# add nodes
template.add_point(label="nose", row=635, col=427)
template.add_point(label="left_eye", row=597, col=404)
template.add_point(label="right_eye", row=685, col=401)
template.add_point(label="left_ear", row=575, col=431)
template.add_point(label="right_ear", row=723, col=425)
template.add_point(label="left_shoulder", row=502, col=614)
template.add_point(label="right_shoulder", row=794, col=621)
template.add_point(label="left_elbow", row=456, col=867)
template.add_point(label="right_elbow", row=837, col=874)
template.add_point(label="left_wrist", row=446, col=1066)
template.add_point(label="right_wrist", row=845, col=1073)
template.add_point(label="left_hip", row=557, col=1035)
template.add_point(label="right_hip", row=743, col=1043)
template.add_point(label="left_knee", row=541, col=1406)
template.add_point(label="right_knee", row=751, col=1421)
template.add_point(label="left_ankle", row=501, col=1760)
template.add_point(label="right_ankle", row=774, col=1765)
# add edges
template.add_edge(src="left_ankle", dst="left_knee")
template.add_edge(src="left_knee", dst="left_hip")
template.add_edge(src="right_ankle", dst="right_knee")
template.add_edge(src="right_knee", dst="right_hip")
template.add_edge(src="left_hip", dst="right_hip")
template.add_edge(src="left_shoulder", dst="left_hip")
template.add_edge(src="right_shoulder", dst="right_hip")
template.add_edge(src="left_shoulder", dst="right_shoulder")
template.add_edge(src="left_shoulder", dst="left_elbow")
template.add_edge(src="right_shoulder", dst="right_elbow")
template.add_edge(src="left_elbow", dst="left_wrist")
template.add_edge(src="right_elbow", dst="right_wrist")
template.add_edge(src="left_eye", dst="right_eye")
template.add_edge(src="nose", dst="left_eye")
template.add_edge(src="nose", dst="right_eye")
template.add_edge(src="left_eye", dst="left_ear")
template.add_edge(src="right_eye", dst="right_ear")
template.add_edge(src="left_ear", dst="left_shoulder")
template.add_edge(src="right_ear", dst="right_shoulder")


settings = {"point_threshold": 0.1}

if not sly.is_production():
    local_bboxes = [
        {"bbox": np.array([245, 72, 411, 375, 1.0])},
        {"bbox": np.array([450, 204, 633, 419, 1.0])},
        {"bbox": np.array([35, 69, 69, 164, 1.0])},
        {"bbox": np.array([551, 99, 604, 216, 1.0])},
        {"bbox": np.array([440, 72, 458, 106, 1.0])},
    ]
    settings["local_bboxes"] = local_bboxes

m = ViTPoseModel(
    use_gui=True,
    custom_inference_settings=settings,
    keypoints_template=template,
)

if sly.is_production():
    m.serve()
else:
    # for local development and debugging without GUI
    selected_model = "ViTPose small with classic decoder"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings)

    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=2)
    print(f"Predictions and visualization have been saved: {vis_path}")
