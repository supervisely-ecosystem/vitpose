
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

![screenshot-dev-supervise-ly-apps-7412-sessions-27716-1676977390333](https://user-images.githubusercontent.com/91027877/220327825-49c9433b-4ac4-45d7-a1c4-cb1e2bfde606.png)

**Step 2.** Wait for the model to deploy

![screenshot-dev-supervise-ly-apps-7412-sessions-27761-1676980517943](https://user-images.githubusercontent.com/91027877/220338410-3812095f-e9d2-4a2a-933e-0579b280ca46.png)

# Related apps

You can use served model in next Supervisely Applications ⬇️

- [Apply Detection and Pose Estimation Models to Images Project](https://dev.supervise.ly/ecosystem/apps/apply-det-and-pose-estim-models-to-project) - app allows to label images project using served  detection and pose estimation models.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det-and-pose-estim-models-to-project" src="https://user-images.githubusercontent.com/97401023/220314584-93315979-c833-4e5b-bc57-6c85a0f7c127.png" width="350px"/>
