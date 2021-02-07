# dog-pose-estimation

Fun project for markerless pose estimation of dogs using DeepLabCut
![](coco_pose.gif)

## Step 1 - Collect Video

## Step 2 - Install DeepLabCut (DLC) Toolkit

## Step 3 - Create DLC Project

### Launch DLC

### Upload Videos

### Define Body Parts

Edit the config.yaml to add body parts. These are the parts that I used for my dog:

```
bodyparts:
  - nose
  - leftear
  - rightear
  - tailstart
  - tailend
  - rightfrontpaw
  - leftfrontpaw
  - rightrearpaw
  - leftrearpaw
  - collar
```

## Step 4 - Label Video Frames

### Extract Frames

### Label Frames

Use the DLC Labeling Toolbox to label frames. DLC will automatically select frames from the video(s) you import using kmeans quantization. It will also downsample the frames before annotation.

## Step 5 - Create Training Dataset

## Step 6 - Train the Model

## Step 7 - Evaluate the Model

## Step 8 - Create Labeled Video
