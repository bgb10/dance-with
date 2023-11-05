# Dance-with: Dance with your friends with the right pose!

-- Picture to be added

`Dance-with` corrects your dance posture using multi-person 2D pose estimation Deep Learning model, OpenPose. The task is to predict a pose: body skeleton, which consists of a predefined set of keypoints and connections between them, and tells you which body parts don't fit between multiple people.

## Motivation

People have become more interested in dancing due to the large number of dancing videos on Instagram. People often take videos of themselves dancing with their friends, but people who are not good dancers are not sure how much their posture is wrong. Dance-with analyzes the posture of the people in the video and shows them which body part are not same between other people, so allows you to dance in sync with each other.

## Features

- Alerts which body parts have inconsistencies between people in your video
- Displays estimated poses in the the resulting frame

## Installation

```bash
# Recommend to turn on python virtualenv before installing the dependencies
pip install -r requirements.txt
pip install ./common/python
pip install -r ./common/python/requirements.txt
pip install -r ./common/python/requirements_ovms.txt
omz_downloader --list models.lst
```
 
## Running

Running the application with the `-h` option yields the following usage message:

```
usage: dance_with.py [-h] -m MODEL -at {ae,hrnet,openpose} -i
                                     INPUT [--loop] [-o OUTPUT]
                                     [-limit OUTPUT_LIMIT] [-d DEVICE]
                                     [-t PROB_THRESHOLD] [--tsize TSIZE]
                                     [-nireq NUM_INFER_REQUESTS]
                                     [-nstreams NUM_STREAMS]
                                     [-nthreads NUM_THREADS] [-no_show]
                                     [--output_resolution OUTPUT_RESOLUTION]
                                     [-u UTILIZATION_MONITORS] [-r]
                                     [-ath NUM_THRESHOLD]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -at {ae,higherhrnet,openpose}, --architecture_type {ae,higherhrnet,openpose}
                        Required. Specify model' architecture type.
  -i INPUT, --input INPUT
                        Required. An input to process. The input must be a
                        single image, a folder of images, video file or camera
                        id.
  --loop                Optional. Enable reading the input in a loop.
  -o OUTPUT, --output OUTPUT
                        Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086
  -limit OUTPUT_LIMIT, --output_limit OUTPUT_LIMIT
                        Optional. Number of frames to store in output. If 0 is
                        set, all frames are stored.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU or
                        GPU is acceptable. The demo
                        will look for a suitable plugin for device specified.
                        Default value is CPU.
 -ath NUM_THRESHOLD, --angle_threshold NUM_THRESHOLD
                        Threshold for checking whether skeleton has similar pose.
                        If the angle of one skeleton is more than a threshold away
                        from the average of the angles of all armature, it is diagnosed as an incorrect pose.

Common model options:
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for poses filtering.
  --tsize TSIZE         Optional. Target input size. This demo implements
                        image pre-processing pipeline that is common to human
                        pose estimation approaches. Image is first resized to
                        some target size and then the network is reshaped to
                        fit the input image shape. By default target image
                        size is determined based on the input shape from IR.
                        Alternatively it can be manually set via this
                        parameter. Note that for OpenPose-like nets image is
                        resized to a predefined height, which is the target
                        size in this case. For Associative Embedding-like nets
                        target size is the length of a short first image side.

Inference options:
  -nireq NUM_INFER_REQUESTS, --num_infer_requests NUM_INFER_REQUESTS
                        Optional. Number of infer requests
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        Optional. Number of streams to use for inference on
                        the CPU or/and GPU in throughput mode (for HETERO and
                        MULTI device cases use format
                        <device1>:<nstreams1>,<device2>:<nstreams2> or just
                        <nstreams>).
  -nthreads NUM_THREADS, --num_threads NUM_THREADS
                        Optional. Number of threads to use for inference on
                        CPU (including HETERO cases).

Input/output options:
  -no_show, --no_show   Optional. Don't show output.
  --output_resolution OUTPUT_RESOLUTION
                        Optional. Specify the maximum output window resolution
                        in (width x height) format. Example: 1280x720.
                        Input frame used by default.
  -u UTILIZATION_MONITORS, --utilization_monitors UTILIZATION_MONITORS
                        Optional. List of monitors to show initially.

Debug options:
  -r, --raw_output_message
                        Optional. Output inference results raw values showing.
```

Running the application with the empty list of options yields the short usage message and an error message.

### Supported Models

* architecture_type=openpose
  * human-pose-estimation-0001
* architecture_type=ae
  * human-pose-estimation-0005
  * human-pose-estimation-0006
  * human-pose-estimation-0007
* architecture_type=higherhrnet
  * higher-hrnet-w32-human-pose-estimation

### Example

```sh
python3 ./dance_with.py \
  -d CPU \
  -i 0 \
  -m ./intel/human-pose-estimation-0005/FP16/human-pose-estimation-0005.xml \
  -at ae
  -r
```

## Implementation
### OpenPose
`OpenPose` is the real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images.

- Use a pre-trained OpenPose model to find each of the 17 joints and then find the 19 skeletons that connect them.
- Show each joint and skeleton in OpenCV

### Intel OpenVINO & Intel Open Model Zoo
`Intel OpenVINO` is a framework that converts various DL framework models into OpenVINO models to ensure compatibility and optimize inference through quantization, pruning, etc. `Intel Open Model Zoo` includes optimized deep learning models and a set of demos to expedite development of high-performance deep learning inference applications. 

- Convert PyTorch-based OpenPose models to OpenVINO models for use

### Skeleton Comparison
- Each person has a Pose, and the Pose has multiple Skeletons.
- For each person, find the direction (i.e. angle) of the Skeleton's vector and average the angle for each Skeleton part.
- Detect and notify outliers if the average of the angles deviates from the mean by more than a threshold(argument value `-ath`, default is 0.5rad).

## Demo

The demo uses OpenCV to display the resulting frame with estimated poses.
The demo reports

* **FPS**: average rate of video frame processing (frames per second).
* **Latency**: average time required to process one frame (from reading the frame to displaying the results).
* Latency for each of the following pipeline stages:
  * **Decoding** — capturing input data.
  * **Preprocessing** — data preparation for inference.
  * **Inference** — infering input data (images) and getting a result.
  * **Postrocessing** — preparation inference result for output.
  * **Rendering** — generating output image.

You can use these metrics to measure application-level performance.

## Further Improvement


## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
