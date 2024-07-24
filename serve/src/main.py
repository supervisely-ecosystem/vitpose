import supervisely as sly
from supervisely.app.widgets import RadioGroup, Field, Table, NotificationBox, Container
from typing_extensions import Literal
from typing import List, Any, Dict, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import torch
import torchvision
import copy
from dotenv import load_dotenv
import cv2


# from mmpose.apis import inference_top_down_pose_model, init_pose_model
try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except ImportError:
    import sys
    from pathlib import Path

    vitpose_path = Path("/ViTPose")
    mmpose_path = vitpose_path.joinpath("mmpose")
    sys.path.insert(0, str(vitpose_path.resolve()))
    sys.path.insert(0, str(mmpose_path.resolve()))

    from mmpose.apis import inference_top_down_pose_model, init_pose_model


import numpy as np
import pandas as pd
import os
from src.keypoints_templates import human_template, animal_template
from src.animal_species import animals


load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
root_source_path = str(Path(__file__).parents[2])
app_source_path = str(Path(__file__).parents[1])
model_data_path = os.path.join(root_source_path, "models", "model_data.json")
configs_path = os.path.join(root_source_path, "configs")
api = sly.Api()


class ViTPoseModel(sly.nn.inference.PoseEstimation):
    def add_content_to_custom_tab(self, gui):
        self.select_task_type = RadioGroup(
            items=[
                RadioGroup.Item(value="human pose estimation"),
                RadioGroup.Item(value="animal pose estimation"),
            ],
            direction="vertical",
        )
        select_task_type_f = Field(self.select_task_type, "Select task type")
        return select_task_type_f

    def add_content_to_pretrained_tab(self, gui):
        animal_table_data = pd.DataFrame(data=animals, columns=[" "])
        animal_species_table = Table()
        animal_species_table.read_pandas(animal_table_data)
        animal_species_table_f = Field(animal_species_table, "List of supported animal species")
        notification = NotificationBox(
            description="""
            Please note that object detection models pretrained on COCO will be able to detect
            only 9 animal species from the list above: cat, dog, horse, sheep, cow, elephant,
            bear, zebra and giraffe. If you will use such detection models, other animal species
            will be ignored, because pose estimation model can't detect animal's keypoints without
            bounding box provided. If you want to perform pose estimation of animal species which
            are not presented in COCO (for example, lions), you have 2 options: 1) train custom
            object detection model to detect lions or 2) use ROI inference mode and draw bounding
            boxes of lions by yourself in annotation tool.

            You should also take into consideration the fact that the list above is not strict and
            such animal pose estimation models are able to estimate pose not only of animal species
            from AP10K dataset (from the list above), but also species related to them. For example, 
            even through ocelot is not in the list of AP10K dataset classes, ViTPose+ models pretrained 
            on AP10K dataset are still able to detect keypoints of ocelots, because ocelot's posture is 
            very similar to other species from the cat family, which are presented in AP10K dataset 
            (like lions, tigers, bobcats, etc.).
            """
        )
        self.custom_animal_content = Container(widgets=[animal_species_table_f, notification])
        self.custom_animal_content.hide()

        @gui._models_table.value_changed
        def on_model_selected(model_row):
            if model_row[0].endswith("animal pose estimation"):
                self.custom_animal_content.show()
            elif (
                model_row[0].endswith("human pose estimation")
                and not self.custom_animal_content.is_hidden()
            ):
                self.custom_animal_content.hide()

        return self.custom_animal_content

    def get_task_type(self):
        if sly.is_production():
            model_source = self.gui.get_model_source()
            if model_source == "Pretrained models":
                selected_model_name = self.gui.get_checkpoint_info()["Model"]
                if selected_model_name.endswith("human pose estimation"):
                    return "human pose estimation"
                elif selected_model_name.endswith("animal pose estimation"):
                    return "animal pose estimation"
            elif model_source == "Custom models":
                return self.select_task_type.get_value()
        else:
            return "human pose estimation"

    def set_template(self):
        if sly.is_production():
            task_type = self.get_task_type()
            if task_type == "human pose estimation":
                self.keypoints_template = human_template
            elif task_type == "animal pose estimation":
                self.keypoints_template = animal_template
        else:
            self.keypoints_template = human_template

    def preprocess_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        experts = dict()
        new_checkpoint = copy.deepcopy(checkpoint)
        state_dict = new_checkpoint["state_dict"]
        for key, value in state_dict.items():
            if "mlp.experts" in key:
                experts[key] = value
        keys = checkpoint["state_dict"].keys()
        target_expert = 0
        new_checkpoint = copy.deepcopy(checkpoint)
        for key in keys:
            if "mlp.fc2" in key:
                value = new_checkpoint["state_dict"][key]
                value = torch.cat(
                    [value, experts[key.replace("fc2.", f"experts.{target_expert}.")]], dim=0
                )
                new_checkpoint["state_dict"][key] = value
        if self.get_task_type() == "human pose estimation":
            torch.save(new_checkpoint, weights_path)
        names = ["aic", "mpii", "ap10k", "apt36k", "wholebody"]
        num_keypoints = [14, 16, 17, 17, 133]
        weight_names = [
            "keypoint_head.deconv_layers.0.weight",
            "keypoint_head.deconv_layers.1.weight",
            "keypoint_head.deconv_layers.1.bias",
            "keypoint_head.deconv_layers.1.running_mean",
            "keypoint_head.deconv_layers.1.running_var",
            "keypoint_head.deconv_layers.1.num_batches_tracked",
            "keypoint_head.deconv_layers.3.weight",
            "keypoint_head.deconv_layers.4.weight",
            "keypoint_head.deconv_layers.4.bias",
            "keypoint_head.deconv_layers.4.running_mean",
            "keypoint_head.deconv_layers.4.running_var",
            "keypoint_head.deconv_layers.4.num_batches_tracked",
            "keypoint_head.final_layer.weight",
            "keypoint_head.final_layer.bias",
        ]
        exist_range = True
        for i in range(5):

            new_checkpoint = copy.deepcopy(checkpoint)

            target_expert = i + 1

            for key in keys:
                if "mlp.fc2" in key:
                    expert_key = key.replace("fc2.", f"experts.{target_expert}.")
                    if expert_key in experts:
                        value = new_checkpoint["state_dict"][key]
                        value = torch.cat([value, experts[expert_key]], dim=0)
                    else:
                        exist_range = False

                    new_checkpoint["state_dict"][key] = value

            if not exist_range:
                break

            for tensor_name in weight_names:
                new_checkpoint["state_dict"][tensor_name] = new_checkpoint["state_dict"][
                    tensor_name.replace("keypoint_head", f"associate_keypoint_heads.{i}")
                ]

            for tensor_name in [
                "keypoint_head.final_layer.weight",
                "keypoint_head.final_layer.bias",
            ]:
                new_checkpoint["state_dict"][tensor_name] = new_checkpoint["state_dict"][
                    tensor_name
                ][: num_keypoints[i]]
            if names[i] == "ap10k" and self.get_task_type() == "animal pose estimation":
                torch.save(new_checkpoint, weights_path)

    def get_models(self, mode="table"):
        model_data = sly.json.load_json_file(model_data_path)
        if mode == "table":
            table_data = model_data.copy()
            for element in table_data:
                del element["config_file_name"]
                del element["weights_link"]
            return table_data
        elif mode == "links":
            models_data_processed = {}
            for element in model_data:
                models_data_processed[element["Model"]] = {
                    "config": element["config_file_name"],
                    "weights": element["weights_link"],
                }
            return models_data_processed

    def get_weights_and_config_path(self, model_dir):
        model_source = self.gui.get_model_source()
        if model_source == "Pretrained models":
            models_data = self.get_models(mode="links")
            selected_model = self.gui.get_checkpoint_info()["Model"]
            weights_link = models_data[selected_model]["weights"]
            weights_file_name = models_data[selected_model]["config"][:-2] + "pth"
            weights_dst_path = os.path.join(model_dir, weights_file_name)
            if not sly.fs.file_exists(weights_dst_path):
                self.download(src_path=weights_link, dst_path=weights_dst_path)
            config_path = os.path.join(
                root_source_path, "configs", models_data[selected_model]["config"]
            )
            return weights_dst_path, config_path
        elif model_source == "Custom models":
            custom_link = self.gui.get_custom_link()
            weights_file_name = os.path.basename(custom_link)
            weights_dst_path = os.path.join(model_dir, weights_file_name)
            if not sly.fs.file_exists(weights_dst_path):
                self.download(
                    src_path=custom_link,
                    dst_path=weights_dst_path,
                )
            config_path = self.download(
                src_path=os.path.join(os.path.dirname(custom_link), "pose_config.py"),
                dst_path=os.path.join(configs_path, "pose_config.py"),
            )
            return weights_dst_path, config_path

    def load_on_device(
        self,
        model_dir,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # set keypoints template
        self.set_template()
        # define model config and checkpoint
        if sly.is_production():
            pose_checkpoint, pose_config = self.get_weights_and_config_path(model_dir)
            if (
                self.gui.get_checkpoint_info()["Model"].startswith("ViTPose+")
                and self.gui.get_model_source() == "Pretrained models"
            ):
                self.preprocess_weights(pose_checkpoint)
        else:
            # for local debug only
            models_data = self.get_models(mode="links")
            weights_link = models_data[selected_model]["weights"]
            weights_file_name = models_data[selected_model]["config"][:-2] + "pth"
            pose_checkpoint = os.path.join(model_dir, weights_file_name)
            if not sly.fs.file_exists(pose_checkpoint):
                sly.fs.download(url=weights_link, save_path=pose_checkpoint)
            pose_config = os.path.join(
                root_source_path, "configs", models_data[selected_model]["config"]
            )
        # initialize pose estimator
        self.pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
        # define class names
        self.task_type = self.get_task_type()
        if self.task_type == "human pose estimation":
            self.class_names = ["person_keypoints"]
        elif self.task_type == "animal pose estimation":
            self.class_names = [animal + "_keypoints" for animal in animals]
            self.class_names.append("animal_keypoints")
        print(f"✅ Model has been successfully loaded on {device.upper()} device")

    def get_classes(self):
        return self.class_names

    def get_info(self):
        info = super().get_info()
        info["videos_support"] = False
        info["async_video_inference_support"] = False
        info["tracking_on_videos_support"] = False
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

        # prepare bounding boxes
        if "local_bboxes" in settings:
            bboxes = settings["local_bboxes"]
        elif "detected_bboxes" in settings:
            bboxes = settings["detected_bboxes"]
            for i in range(len(bboxes)):
                box = bboxes[i]["bbox"]
                bboxes[i] = {"bbox": np.array(box)}
        else:
            bboxes = bbox

        # prepare class names
        if "detected_classes" in settings:
            detected_classes = [cls + "_keypoints" for cls in settings["detected_classes"]]

        # inference pose estimator
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
        for i, result in enumerate(pose_results):
            included_labels, included_point_coordinates = [], []
            point_coordinates, point_scores = result["keypoints"][:, :2], result["keypoints"][:, 2]
            for j, (point_coordinate, point_score) in enumerate(
                zip(point_coordinates, point_scores)
            ):
                if point_score >= point_threshold:
                    included_labels.append(point_labels[j])
                    included_point_coordinates.append(point_coordinate)
            if self.task_type == "human pose estimation":
                class_name = "person_keypoints"
            elif self.task_type == "animal pose estimation":
                class_name = None
                if "detected_classes" in settings:  # for usage in combination with detector
                    if detected_classes[i] in self.class_names:
                        class_name = detected_classes[i]
                elif "rectangle" in settings:  # for ROI inference mode
                    rectangle_data = settings["rectangle"]
                    objclass_info = api.object_class.get_info_by_id(id=rectangle_data["classId"])
                    if objclass_info.name + "_keypoints" in self.class_names:
                        class_name = objclass_info.name + "_keypoints"
                if class_name is None:  # for case when none of the conditions above were met
                    class_name = "animal_keypoints"
            if len(included_labels) > 1:
                results.append(
                    sly.nn.PredictionKeypoints(
                        class_name, included_labels, included_point_coordinates
                    )
                )
        return results


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
)

if sly.is_production():
    m.serve()
else:
    # for local development and debugging without GUI
    selected_model = "ViTPose small classic for human pose estimation"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    m.load_on_device(m.model_dir, device)
    image_path = "./demo_data/image_01.jpg"
    results = m.predict(image_path, settings)

    vis_path = "./demo_data/image_01_prediction.jpg"
    m.visualize(results, image_path, vis_path, thickness=2)
    print(f"Predictions and visualization have been saved: {vis_path}")
