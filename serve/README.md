
<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/97401023/220314920-2c2892eb-c11b-4fea-a17e-898a09fcfbed.png"/>
  
# Serve ViTPose
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
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

Serve ViTPose model as Supervisely Application. ViTPose is an open source pose estimation toolbox based on PyTorch. Learn more about ViTPose and available models [here](https://github.com/open-mmlab/mmsegmentation).

Model serving allows to apply model to image (URL, local file, Supervisely image id) with 2 modes (full image, image ROI). Also app sources can be used as example how to use downloaded model weights outside Supervisely.

Application key points:
- Serve custom and MMSegmentation models
- Deployed on GPU
