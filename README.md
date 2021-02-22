# 🐕 dog-pose-estimation
<img src="coco_pose3.gif" width="240"> <img src="coco_pose.gif" width="240"> <img src="coco_pose2.gif" width="240">

As a fun weekend project, I developed a pose estimation model with my dog using [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut). DLC is a free and open source toolkit that enables researchers to estimate the pose of a subject. DLC leverages transfer learning, which is the ability to take a fully trained neural network model — which was trained with a large supervised data set — and re-train it with a smaller set of images.  

Using the DLC Python toolkit, I was able to collect and label ~100 images of my dog Coco to re-train an accurate pose estimation model without needing a GPU or extensive programming experience. The toolkit made it easy to extract videos taken on my iPhone, annotate frames, train a model with TensorFlow, analyze and refine the model, and generate labeled videos that track Coco's pose with markers and show an outline of her skeleton.  


📖 Read: [Nath*, Mathis* et al., 2019](https://www.biorxiv.org/content/10.1101/476531v1)  
📺 Watch: [Mackenzie Mathis: DeepLabCut Tech Demo](https://youtu.be/iKazf5F7bPU)  
💻 Try DLC: https://github.com/DeepLabCut/DeepLabCut    


## 🎬 Step 1 - Collect Video  
The first step is to collect videos of your dog. I took both indoor and ourdoor videos of Coco from many different angles with my iPhone X. Because I eventually want to detect her position based on pose (standing, sitting and down-stay), I took videos of each position. 

All of the videos used for this experiment are available to download on [Google Drive](https://drive.google.com/drive/folders/1mNyxVW1pWvxEzxDhhHx2TqU558UwnZbF?usp=sharing).

<b>Tip:</b> Cameras should be placed such that the features you want to track are visible.  

## 💻 Step 2 - Install DeepLabCut    
For this experiment, I used my Ubuntu 20.04 desktop computer with an Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz and 128GB of RAM (🚫 No GPU required 🚫). Only ~30GB of RAM was used during the training, so 128GB is overkill.   

To set up my environment, I used the [conda file for CPU](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/DLC-CPU.yaml) provided by DeepLabCut to create an environment called `DLC-CPU`. This environment will use `python=3.7` `tensorflow=1.13.1` and `wxPython<4.1.0`. There is also a conda file for GPU, which uses `tensorflow-gpu=1.13.1`. See more detailed instructions in the [DLC docs](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/README.md) 

1. Install [Anaconda 3](https://www.anaconda.com/distribution/) if you don't have it already
2. Download DLC's [conda file for CPU](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/DLC-CPU.yaml)
3. Open terminal where you placed the file and run `conda env create -f DLC-CPU.yaml`

## ✏️ Step 3 - Create and Configure Project  

### Create the Project

```python
deeplabcut.create_new_project('ProjectName','YourName', ['/Users/FullPath/OfVideo1.mov', '/Users/FullPath/OfVideo2.mov', '/Users/FullPath/OfVideo1.mov'],
              copy_videos=True, multianimal=False)
```

### Define Body Parts

Edit the config.yaml to add body parts. These are the parts that I used for my dog:

```
bodyparts:
- nose
- leftear
- rightear
- collar
- spine1
- spine2
- spine3
- tailstart
- tailend
- frontleftleg
- frontleftpaw
- frontrightleg
- frontrightpaw
- backleftleg
- backleftpaw
- backrightleg
- backrightpaw
```

### Set config_path

For the next steps, you will use the environment variable `config_path`.

```python
config_path = '/Users/FullPath/DeepLabCut/yourprojectname/config.yaml'
```

## 🖌️ Step 4 - Label Video Frames 

![](label_frame.gif)

### Select Frames to Label:

```python
deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', crop=False)
```

### Label Frames

Use the DLC Labeling Toolbox to label frames. DLC will automatically select frames from the video(s) you import using kmeans quantization. It will also downsample the frames before annotation.

### Build Skeleton

## 📁 Step 5 - Create Training Dataset  
```python
deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')
```

## ⚙️ Step 6 - Train the Model  
```python
deeplabcut.train_network(config_path, shuffle=1, trainingsetindex=0, gputouse=None, max_snapshots_to_keep=5, autotune=False, displayiters=100, saveiters=15000, maxiters=30000, allow_growth=True)
```

## 📊 Step 7 - Evaluate the Model  

## 📽️ Step 8 - Create Labeled Video  
