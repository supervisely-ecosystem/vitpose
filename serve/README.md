
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

If you select model for animal pose estimation, you will also see list of supported animal species and basic information about pitfalls of animal pose estimation:

![screenshot-dev-supervise-ly-apps-7412-sessions-28159-1677553835357](https://user-images.githubusercontent.com/91027877/221749128-99812eca-30ae-48ef-b43f-ce73b92cd471.png)

**Custom models**

Copy model file path from Team Files and select task type:

![custom_models](https://user-images.githubusercontent.com/91027877/221755821-088d7de3-1297-4c87-856c-75fa75f973f8.gif)

**Example**

Step 1. Scroll apps field up to get access to apps for labeling UI:

![step_1_1](https://user-images.githubusercontent.com/91027877/222123233-8b063e55-3263-4f49-ba26-aa7586400eab.jpeg)

![step_1_2](https://user-images.githubusercontent.com/91027877/222129713-66e97726-c0ba-4e2f-89bc-f22b0f221c59.gif)

Step 2. Select "NN Image Labeling" app and run it:

![step_2_1](https://user-images.githubusercontent.com/91027877/222123429-a65ed482-53aa-4056-9d9e-340f52c6c8dd.jpeg)

![step_2_2](https://user-images.githubusercontent.com/91027877/222123616-954a33c2-8774-40a0-90bf-601ed52efd83.jpeg)

Step 3. Connect to ViTPose:

![step_3_1](https://user-images.githubusercontent.com/91027877/222124338-5b35b061-a444-4dc2-959e-2c50f6420809.jpeg)

![step_3_2](https://user-images.githubusercontent.com/91027877/222124496-31850d79-201a-4bd8-a6f0-d60f54c4727c.jpeg)

Step 4. Create rectangle class and put target object in it:

![step_4_1](https://user-images.githubusercontent.com/91027877/222125562-22d2963f-0e14-49bb-a840-66e3af15cfb3.jpeg)

![step_4_2](https://user-images.githubusercontent.com/91027877/222125603-43c45d79-4d60-4223-a548-c296e514e963.jpeg)

![step_4_3](https://user-images.githubusercontent.com/91027877/222132268-f68bef13-b6b2-41fa-998c-9ec4836db015.gif)

Step 5. Click on "Apply model to ROI":

![step_5](https://user-images.githubusercontent.com/91027877/222125951-cc5abd83-af30-4a32-bb1b-e6b169404505.jpeg)

Result:

![step_result](https://user-images.githubusercontent.com/91027877/222126060-071928cb-e5d8-4485-a9c5-2b234b61bec8.png)

For animal pose estimation task: if you create bounding box with class name, which is presented in the list of supported animal species, keypoints skeleton class with name "{your_class_name}_keypoints" will be created. Otherwise keypoints skeleton class with name "animal_keypoints" will be created.

If there is only a part of target object on the image, then you can tune point threshold in app settings to get rid of unnecessary points:

https://user-images.githubusercontent.com/91027877/222139867-61d01ad2-d576-48ee-aae0-990c014c709e.mp4



# Related apps

You can use served model in next Supervisely Applications ⬇️

- [Apply Detection and Pose Estimation Models to Images Project](https://dev.supervise.ly/ecosystem/apps/apply-det-and-pose-estim-models-to-project) - app allows to label images project using served  detection and pose estimation models.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det-and-pose-estim-models-to-project" src="https://user-images.githubusercontent.com/97401023/220315624-c6e79003-39fb-43e7-be48-ead1c9fae771.png" width="350px"/>
    
# Acknowledgment

This app is based on the great work `ViTPose` ([github](https://github.com/ViTAE-Transformer/ViTPose)). ![GitHub Org's stars](https://img.shields.io/github/stars/ViTAE-Transformer/ViTPose?style=social)
