
<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/97401023/220314920-2c2892eb-c11b-4fea-a17e-898a09fcfbed.png"/>
  
# Serve ViTPose
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#example-apply-vitpose-to-image-in-labeling-tool">Example: apply ViTPose to image in labeling tool</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>
  
[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../../supervisely-ecosystem/vitpose/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/vitpose)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/vitpose/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/vitpose/serve.png)](https://supervisely.com)

</div>

# Overview

Serve ViTPose model as Supervisely Application. ViTPose is an open source pose estimation toolbox based on PyTorch. Learn more about ViTPose and available models [here](https://github.com/ViTAE-Transformer/ViTPose).

Model serving allows to apply model to image (URL, local file, Supervisely image id) with 2 modes (in combination with detection model using Apply Detection and Pose Estimation Models to Images Project app or image ROI). Also app sources can be used as example how to use downloaded model weights outside Supervisely.

# How to Run

**Pretrained models**

**Step 1.** Select pretrained model architecture and press the **Serve** button

![screencapture-dev-supervise-ly-apps-7412-sessions-28261-2023-03-01-17_57_13](https://user-images.githubusercontent.com/91027877/222177351-2047f406-f6e3-4ba4-a395-73f2b3c5fcd9.png)

If you select model for animal pose estimation, you will also see list of supported animal species and basic information about pitfalls of animal pose estimation:

![screencapture-dev-supervise-ly-apps-7412-sessions-28261-2023-03-01-18_00_09](https://user-images.githubusercontent.com/91027877/222177825-bc0da3cf-7c06-447a-bf81-e627f94614eb.png)

**Step 2.** Wait for the model to deploy

![screencapture-dev-supervise-ly-apps-7412-sessions-28261-2023-03-01-18_02_23](https://user-images.githubusercontent.com/91027877/222178391-c32985b6-74a0-46dc-a671-dc95c9532c34.png)

**Custom models**

Copy model file path from Team Files and select task type:

https://user-images.githubusercontent.com/91027877/222154098-6ce825eb-ec32-4c2b-ae3b-545f0ec288d1.mp4

# Example: apply ViTPose to image in labeling tool

Run NN Image Labeling app, connect to ViTPose, create bounding box and click on "Apply model to ROI":

https://user-images.githubusercontent.com/91027877/240876681-c7547f2d-643a-4d9a-899c-db569b255bf7.mp4

For animal pose estimation task: if you create bounding box with class name, which is presented in the list of supported animal species, keypoints skeleton class with name "{your_class_name}_keypoints" will be created. Otherwise keypoints skeleton class with name "animal_keypoints" will be created.

If you want keypoints labels to be shown, go to image settings and set parameter "Show keypoints labels" to "Always". You can also tune line width and many other visualization parameters:

https://user-images.githubusercontent.com/91027877/240876898-2dcc1542-8338-4778-aaab-55d81054814d.mp4

If there is only a part of target object on the image, then you can increase point threshold in app settings to get rid of unnecessary points:

https://user-images.githubusercontent.com/91027877/222139867-61d01ad2-d576-48ee-aae0-990c014c709e.mp4

# Related apps

You can use served model in next Supervisely Applications ⬇️

- [Apply Detection and Pose Estimation Models to Images Project](../../../../supervisely-ecosystem/apply-det-and-pose-estim-models-to-project) - app allows to label images project using served  detection and pose estimation models.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det-and-pose-estim-models-to-project" src=https://user-images.githubusercontent.com/91027877/222169346-6c813d3a-6216-44da-bff1-98654943398b.png width="650px"/>
    
# Acknowledgment

This app is based on the great work `ViTPose` ([github](https://github.com/ViTAE-Transformer/ViTPose)). ![GitHub Org's stars](https://img.shields.io/github/stars/ViTAE-Transformer/ViTPose?style=social)
