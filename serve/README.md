
<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/97401023/220314920-2c2892eb-c11b-4fea-a17e-898a09fcfbed.png"/>
  
# Serve ViTPose
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Example:-apply-ViTPose-to-image-in-labeling-tool">Example: apply ViTPose to image in labeling tool</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>
  
[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/vitpose/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/vitpose)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mmsegmentation/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mmsegmentation/serve.png)](https://supervise.ly)

</div>

# Overview

Serve ViTPose model as Supervisely Application. ViTPose is an open source pose estimation toolbox based on PyTorch. Learn more about ViTPose and available models [here](https://github.com/ViTAE-Transformer/ViTPose).

Model serving allows to apply model to image (URL, local file, Supervisely image id) with 2 modes (full image, image ROI). Also app sources can be used as example how to use downloaded model weights outside Supervisely.

# How to Use

**Pretrained models**

**Step 1.** Select pretrained model architecture and press the **Serve** button

![image](https://user-images.githubusercontent.com/91027877/221755507-9403d2d1-2062-43a7-bd89-f2d74ca0a48f.png)


**Step 2.** Wait for the model to deploy

![image](https://user-images.githubusercontent.com/91027877/221755657-4a573c23-f478-4167-bbce-7c93de11a330.png)

If you select model for animal pose estimation, you will also see list of supported animal species and basic information about pitfalls of animal pose estimation:

![screenshot-dev-supervise-ly-apps-7412-sessions-28159-1677553835357](https://user-images.githubusercontent.com/91027877/221749128-99812eca-30ae-48ef-b43f-ce73b92cd471.png)

**Custom models**

Copy model file path from Team Files and select task type:

https://user-images.githubusercontent.com/91027877/222154098-6ce825eb-ec32-4c2b-ae3b-545f0ec288d1.mp4

# Example: apply ViTPose to image in labeling tool

Run NN Image Labeling app, connect to ViTPose, create bounding box and click on "Apply model to ROI":

https://user-images.githubusercontent.com/91027877/222157699-6af2fd7b-d90b-40c7-bbb5-4626cb69e696.mp4

For animal pose estimation task: if you create bounding box with class name, which is presented in the list of supported animal species, keypoints skeleton class with name "{your_class_name}_keypoints" will be created. Otherwise keypoints skeleton class with name "animal_keypoints" will be created.

If you want keypoints labels to be shown, go to image settings and set parameter "Show keypoints labels" to "Always". You can also tune line width and many other visualization parameters:

https://user-images.githubusercontent.com/91027877/222158310-49f4a695-5e8f-41e3-aa08-229a9bb6410a.mp4

If there is only a part of target object on the image, then you can increase point threshold in app settings to get rid of unnecessary points:

https://user-images.githubusercontent.com/91027877/222139867-61d01ad2-d576-48ee-aae0-990c014c709e.mp4

# Related apps

You can use served model in next Supervisely Applications ⬇️

- [Apply Detection and Pose Estimation Models to Images Project](https://dev.supervise.ly/ecosystem/apps/apply-det-and-pose-estim-models-to-project) - app allows to label images project using served  detection and pose estimation models.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det-and-pose-estim-models-to-project" src="https://user-images.githubusercontent.com/97401023/220315624-c6e79003-39fb-43e7-be48-ead1c9fae771.png" width="350px"/>
    
# Acknowledgment

This app is based on the great work `ViTPose` ([github](https://github.com/ViTAE-Transformer/ViTPose)). ![GitHub Org's stars](https://img.shields.io/github/stars/ViTAE-Transformer/ViTPose?style=social)
