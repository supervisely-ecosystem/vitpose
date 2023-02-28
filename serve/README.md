
<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/97401023/220314920-2c2892eb-c11b-4fea-a17e-898a09fcfbed.png"/>
  
# Serve ViTPose
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
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

If you selected model for animal pose estimation, you will also see list of supported animal species and basic information about pitfalls of animal pose estimation:

![screenshot-dev-supervise-ly-apps-7412-sessions-28159-1677553835357](https://user-images.githubusercontent.com/91027877/221749128-99812eca-30ae-48ef-b43f-ce73b92cd471.png)

**Custom models**

Copy model file path from Team Files and select task type:

![custom_models](https://user-images.githubusercontent.com/91027877/221755821-088d7de3-1297-4c87-856c-75fa75f973f8.gif)

**Example**

To label your image using ViTPose, run [NN Image Labeling](https://dev.supervise.ly/ecosystem/apps/nn-image-labeling/annotation-tool) app, connect to served ViTPose model, draw bounding box and click "apply model to object ROI":

![roi_inference](https://user-images.githubusercontent.com/91027877/221756280-10cdeea1-db3f-403d-89b5-0a2a34df9021.gif)


# Related apps

You can use served model in next Supervisely Applications ⬇️

- [Apply Detection and Pose Estimation Models to Images Project](https://dev.supervise.ly/ecosystem/apps/apply-det-and-pose-estim-models-to-project) - app allows to label images project using served  detection and pose estimation models.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det-and-pose-estim-models-to-project" src="https://user-images.githubusercontent.com/97401023/220314584-93315979-c833-4e5b-bc57-6c85a0f7c127.png" width="350px"/>
    
# Acknowledgment

This app is based on the great work `ViTPose` ([github](https://github.com/ViTAE-Transformer/ViTPose)). ![GitHub Org's stars](https://img.shields.io/github/stars/ViTAE-Transformer/ViTPose?style=social)
