# Real-Time Human Pose Estimation and Squat Detection using NVIDIA DeepStream

## System Architecture Overview

### 1. Overview
This system is designed for real-time human pose estimation and squat detection using NVIDIA DeepStream. It processes video streams, detects human keypoints, calculates joint angles, and visually overlays results on the output stream. The architecture leverages GStreamer for multimedia handling and DeepStream for AI-based inference.

### 2. Components
#### 2.1 Input Handling
- **GStreamer Pipeline:** The pipeline processes video streams and passes frames through DeepStream’s inference engine.
- **URIDecodeBin:** Handles RTSP or file-based input and ensures compatibility with NVIDIA’s accelerated video decoding.
- **NvStreamMux:** Combines multiple input streams into a batch for efficient processing.

#### 2.2 DeepStream Inference
- **PGIE (Primary GIE):** Runs AI inference using a DeepStream model to detect human keypoints.
- **Tracker (NvTracker):** Assigns unique IDs to detected individuals to maintain tracking consistency across frames.

#### 2.3 Pose Processing and Squat Detection
- **Keypoint Parsing:** Extracts keypoints from metadata and normalizes them.
- **Angle Calculation:** Computes joint angles (knee, hip, and body alignment) to determine squat posture.
- **Squat Counting Logic:** Tracks squat events based on predefined angle thresholds.

#### 2.4 Visualization and Metadata Handling
- **Bounding Boxes & Labels:** Overlays bounding boxes and squat count for each detected person.
- **Pose Skeleton Overlay:** Draws human pose skeletons by connecting keypoints.
- **GStreamer Display Output:** Renders processed frames with metadata annotations.

### 3. Performance & Optimization
- **GPU Acceleration:** Leverages CUDA for real-time inference and visualization.
- **FPS Calculation:** Monitors real-time performance using a frame rate tracking mechanism.
- **Pipeline Optimization:** Uses NVMM memory for efficient memory handling in DeepStream.

## How to Start the Program

### 1. Setup Environment
```sh
export CUDA_VER=12.1
make -C nvdsinfer_custom_impl_Yolo_pose
```

### 2. Run the Application
```sh
xhost +
docker run -it --rm --gpus all -e DISPLAY=$DISPLAY --ipc=host --net=host --privileged -w /app -v $(pwd):/app nvcr.io/nvidia/deepstream:6.3-gc-triton-devel /bin/bash
pip3 install pyds-1.1.8-py3-none-linux_x86_64.whl
```

### 3. Run Inference
#### Start Database:
```sh
docker run -d --name mongodb -p 27017:27017  mongo
```

### 4. Run Inference
#### For multiple-person inference:
```sh
python3 deepstream.py -s file:///app/multiple_person.mp4 -c config_infer_primary_yoloV8_pose.txt
```

#### For single-person inference:
```sh
python3 deepstream.py -s file:///app/single_person.mp4 -c config_infer_primary_yoloV8_pose.txt -w 1080
```

## Configuration Options

### Change the Source
```sh
-s file:// or rtsp:// or http://
--source file:// or rtsp:// or http://
```

### Change the Config Infer File
```sh
-c config_infer.txt
--config-infer config_infer.txt
```

### Change nvstreammux Batch Size (default: 1)
```sh
-b 2
--streammux-batch-size 2
```

### Change nvstreammux Width (default: 1920)
```sh
-w 1280
--streammux-width 1280
```

### Change nvstreammux Height (default: 1080)
```sh
-e 720
--streammux-height 720
```

### Change GPU ID (default: 0)
```sh
-g 1
--gpu-id 1
```

### Change FPS Measurement Interval (default: 5)
```sh
-f 10
--fps-interval 10
```

## NMS Configuration
- The `nms-iou-threshold` is fixed to `0.45`.
- Ensure `cluster-mode=4` in the config_infer file.

## Detection Threshold Configuration
```ini
[class-attrs-all]
pre-cluster-threshold=0.25
topk=300
```

## Assets
Download required assets from the following link:
[Assets Download](https://drive.google.com/file/d/1qIlKyNbVnGxZ5_HWGRC4pa-wwLs9-49_/view?usp=drive_link)

## References
1. [MathWorks: DL-Based Human Pose Estimation for Squat Analysis](https://www.mathworks.com/help/vision/ug/dl-based-human-pose-estimation-for-squat-analysis.html)
2. [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
3. [DeepStream-YOLO Pose Documentation](https://github.com/marcoslucianops/DeepStream-Yolo-Pose/blob/master/docs/YOLOv8_Pose.md)

---
**Note:** The TensorRT engine file may take a long time to generate (more than 10 minutes).


