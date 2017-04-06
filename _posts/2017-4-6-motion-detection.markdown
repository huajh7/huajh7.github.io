---
layout:     post
title:    "运动检测 (Motion Detection)"
subtitle:   "运动侦测，移动侦测，移动检测"
author:     "huajh7"
catalog:    true
header-img: "img/post-bg-universe.jpg"
tags:
  - Motion Detection
date: 2017-4-6  8:40:25
---
# 运动检测 (Motion Detection)

> 又名：运动侦测，移动侦测，移动检测 

## 方法和思路  

### 1. 帧差分法 (frame differencing) 
> Frame differencing is a technique where the computer checks the differencebetween two video frames. If the pixels have changed there apparently was something changing in the image (moving for example). Most techniques work with some blur and threshold, to distict real movement from noise. Because frame could differ too when light conditions in a room change ( and **camera auto focus**, **brightness correction** etc. ). 

> from: [kasperkamperman-computervision-framedifferencing](http://www.kasperkamperman.com/blog/computer-vision/computervision-framedifferencing/)

* 帧间差分法
* 三帧差分法

> 背景差分法(Background difference) : 视频帧图像与背景模型图像进行差分和阈值分割
> 
> 帧差分法： 视频中的一帧图像与另一帧图像进行差分运算

`comment`

>  帧差可说是最简单的一种背景模型，指定视频中的一幅图像为背景，用当前帧与背景进行比较，根据需要过滤较小的差异 （阈值），得到的结果就是前景了

### 2. 背景减除法 (Background subtraction) 

**定义:** 

**Background subtraction algorithm** is to distinguish moving objects (hereafter referred to as the `foreground`) from static, or slow moving, parts
of the scene (called `background`).

> Background subtraction, also known as `foreground detection(前景检测)`, is a technique in the fields of image processing and computer vision wherein an image's foreground is extracted for further processing (object recognition etc.). Generally an image's regions of interest are objects (humans, cars, text etc.) in its foreground. After the stage of image preprocessing (which may include image denoising, post processing like morphology etc.) object localisation is required which may make use of this technique.
> 
>  from [wiki/Background_subtraction](https://en.wikipedia.org/wiki/Background_subtraction)

**需要解决的问题：**

+ **light changes(光照)**: the background model should adapt to gradual or fast `illumination changes` (changing time of day, clouds, etc);
+ **moving background** or **high frequency background objects**(树叶等): the background model should include changing background that is not of interest for visual surveillance, such as `waving trees or branches`;
+ **cast shadows(阴影)**: the background model should include the `shadow cast by moving objects` that apparently behaves itself moving, in order to have a more accurate detection of the moving objects shape;
+ **bootstrapping(初始化)**: the background model should be properly set up even in the absence of a complete and static (free of moving objects) training set at the beginning of the sequence;
+ **camouflage(背景相似)**: moving objects should be detected even if their chromatic features are similar to those of the background model.
+ **motion changes** (camera oscillations);
+ changes in the **background geometry** (e.g., parked cars).
+ **Ghost**区域：当一个原本静止的物体开始运动，背静差检测算法可能会将原来该物体所覆盖的区域错误的检测为运动的，这块区域就成为Ghost.

**技术**：
`Pixel-based` background subtraction: a static background frame, the (weighted) running average [21], first-order low-pass filtering [22], temporal median filtering [23], [24], and the modeling of each pixel with a Gaussian [25]–[27].

需要考虑三个问题：

* 如何 **建立和使用** 背景模型
* 如何 **初始化** 背景模型
* 如何 **实时更新** 背景模型


#### 2.1 高斯模型 (Gaussian Model)

**Single Gussian & Running Gaussian average**

+ Wren, Pfinder: **Real-time tracking of the human body**, `1997`, cited by `5000+`

> *Abstract* -- Pfinder is a real-time system for tracking people and interpreting thier behavior. It runs at 10Hz on a standard SGI Indy computer, and has performed reliably on thousands of people in many different physical locations. The system uses a multi-class statistical model of color and shape to obtain a 2-D representation of head and hands in a wide range of viewing conditions. Pfinder has been successfully used in a wide range of applications including wireless interfaces, video databases, and low-bandwidth coding.

**Mixture of Gaussian Model**

+ KaewTraKulPong, **An improved adaptive background mixture model for real-time tracking with shadow detection**, `2001`, cited by `1400+`

> *Abstract* -- Real-time segmentation of moving regions in image sequences is a fundamental step in many vision systems including automated visual surveillance, human-machine interface, and very low-bandwidth telecommunications. A typical method is background subtraction. Many background models have been introduced to deal with different problems. One of the successful solutions to these problems is to use a multi-colour background model per pixel proposed by Grimson et al [1,2,3]. However, the method suffers
from slow learning at the beginning, especially in busy environments. In addition, it can not distinguish between moving shadows and moving objects. This paper presents a method which improves this adaptive background mixture model. By reinvestigating the update equations, we utilise different equations at different phases. This allows our system learn faster and more accurately as well as adapt effectively to changing environments. A shadow detection scheme is also introduced in this paper. It is based on a computational colour space that makes use of our background model. A comparison has been made
between the two algorithms. The results show the speed of learning and the accuracy of the model using our update algorithm over the Grimson et al’s tracker. When incorporate with the shadow detection, our method results in far better segmentation than that of Grimson et al.

+ 李鸿, **基于混合高斯模型的运动检测及阴影消除算法研究**, 中国民航大学硕士论文， `2013`, cited by `10`
+ 卢章平，**背景差分与三帧差分结合的运动目标检测算法** ，计算机测量与控制，`2013`, cited by `44`

`comment`
> 混合高斯在现有的背景建模算法中应该算是比较好的，很多新的算法或改进的算法都是基于它的一些原理的不同变体，但混合高斯算法的缺点是计算量相对比较大，速度偏慢，对光照敏感

#### 2.2 W4 algorithm (What? Where? Who? When?)

+ Ismail Haritaoglu, **W4: A Real Time System for Detecting and Tracking People**, `1998`, cited by `1100+`

> *Abstract*  W^4 is a real time visual surveil lance system for detecting and tracking people and monitoring their activities in an outdoor environment. It operates on monocular grayscale video imagery, or on video imagery from an infrared camera. Unlike many of systems for tracking people, W^4 makes no use of color cues. Instead, W^4 employs a combination of shape analysis and tracking to locate people and their parts (head, hands, feet, torso) and to create models of people's appearance so that they can be tracked through
interactions such as occlusions. W^4 is capable of simultaneously tracking multiple people even with occlusion. It runs at 25 Hz for 320x240 resolution images on a dual-pentium PC.

`comment`
> W4算法应该是最早被用于实际应用的一个算法.

#### 2.3 基于颜色信息的背景建模 (color)

+ Horprasert, **A statistical approach for real-time robust background subtraction and shadow detection**, `1999`, cited by `1200+`

> *Abstract* This paper presents a novel algorithm for detecting moving objects from a static background scene that contains shading and shadows using color images. We develop a robust and efficiently computed background subtraction algorithm that is able to cope with local il lumination changes,
such as shadows and highlights, as wel l as global il lumination changes. The algorithm is based on a proposed computational color model which separates the brightness from the chromaticity component. We have applied this method to real image sequences of both indoor and outdoor scenes. The results, which demonstrate the system's performance, and some speed up techniques we employed in our implementation are also shown.

`comment`

> 算法初衷：解决关于全局或局部的光照变化问题，例如阴影和高亮
> 
> 基于颜色信息的背景建模方法,简称Color算法，该算法将像素点的差异分解成Chromaticity差异和Brightness差异，对光照具有很强的鲁棒性，并有比较好的效果，计算速度也比较快，基本可以满足实时性的要求，做了许多视频序列的检测，效果比较理想；

#### 2.4 本征背景法

+ Nuria M. Oliver, **A Bayesian computer vision system for modeling human interactions**, `2000`, cited by `1500+`

> *Abstract* — We describe a real-time computer vision and machine learning system for modeling and recognizing human behaviors in a visual surveillance task [1]. The system is particularly concerned with detecting when interactions between people occur and classifying the type of interaction. Examples of interesting interaction behaviors include following another person, altering one’s path to meet another, and so forth. Our system combines top-down with bottom-up information in a closed feedback loop, with both
components employing a statistical Bayesian approach [2]. We propose and compare two different state-based learning architectures, namely, HMMs and CHMMs for modeling behaviors and interactions. The CHMM model is shown to work much more efficiently and accurately. Finally, to deal with the problem of limited training data, a synthetic “Alife-style” training system is used to develop flexible prior models for recognizing human interactions. We demonstrate the ability to use these a priori models to accurately classify real human behaviors and interactions with no additional tuning or training.

`comment`
> 基于贝叶斯框架 

####  2.5 核密度估计方法

+ Ahmed Elgammal, **Non-parametric model for background subtraction**, `2000`, cited by `2500+`

> *Abstract* Background subtraction is a method typically used to segment moving regions in image sequences taken from a static camera
by comparing each new frame to a model of the scene background. We
present a novel non-parametric background model and a background
subtraction approach. The model can handle situations where the background of the scene is cluttered and not completely static but contains
small motions such as tree branches and bushes. The model estimates
the probability of observing pixel intensity values based on a sample of
intensity values for each pixel. The model adapts quickly to changes in
the scene which enables very sensitive detection of moving targets. We
also show how the model can use color information to suppress detection of shadows. The implementation of the model runs in real-time for
both gray level and color imagery. Evaluation shows that this approach
achieves very sensitive detection with very low false alarm rates.

`comment`
> 比较鲁棒的算法，无需设置参数.

#### 2.6 背景统计模型

> 对一段时间的背景进行统计，然后计算其统计数据（例如平均值、平均差分、标准差、均值漂移值等等），将统计数据作为背景的方法。

**统计平均法**

+ BPL Lo, **Automatic congestion detection system for underground platform**, `2001`, cited by `300+`

> *Abstract* - An automatic monitoring system is proposed in this paper for detecting overcrowding conditions in the platforms of underground train services.
Whenever overcrowding is detected, the system will notify the station operators to take appropriate actions to prevent accidents, such as
people falling off or being pushed onto the tracks. The system is designed to use existing closed circuit television (CCTV) cameras for acquiring
images of the platforms. In order to focus on the passengers on the platform, background subtraction and update techniques are used. In addition, due to the high variation of brightness on the platforms, a variance filter is introduced
to optimize the removal of background pixels. A multi-layer feed forward neural network was developed for classifying the levels of congestion. The system was tested with recorded video from the London Bridge station, and the testing results were shown to be accurate in identifying overcrowding conditions for the unique platform environment. 

**中值滤波法 (Temporal Median filter)**

+ R Cucchiara, **Detecting Moving Objects, Ghosts, and Shadows in Video Streams**, `2003`, cited by `1600+`

> *Abstract* — Background subtraction methods are widely exploited for moving
object detection in videos in many applications, such as traffic monitoring, human motion capture, and video surveillance. How to correctly and efficiently model and update the background model and how to deal with shadows are two of the most distinguishing and challenging aspects of such approaches. This work proposes a general-purpose method that combines statistical assumptions with the objectlevel knowledge of moving objects, apparent objects (ghosts), and shadows acquired in the processing of the previous frames. Pixels belonging to moving objects, ghosts, and shadows are processed differently in order to supply an object-based selective update. The proposed approach exploits color information for both background subtraction and shadow detection to improve object segmentation and background update. The approach proves fast, flexible, and precise in terms of both pixel accuracy and reactivity to background changes.

`comment`

> 
> 统计平均法和中值滤波法，算法的应用具有很大的局限性，只能算是理论上的一个补充.

#### 2.7 复杂背景下的前景物体检测 (FGD)

+ Liyuan Li, **Foreground Object Detection from Videos Containing Complex Background**, `2003`, cited by `500+`

> *Abstract* --  This paper proposes a novel method for detection and segmentation of foreground objects from a video which contains both stationary and moving background objects and undergoes both gradual and sudden “once-off” changes. A Bayes decision rule for classification of background and foreground
from selected feature vectors is formulated. Under this rule, different types of background objects will be classified from foreground objects by choosing a proper feature vector. The stationary background object is described by the color feature, and the moving background object is represented by the color co-occurrence feature. Foreground objects are extracted by fusing the classification results from both stationary and moving pixels. Learning strategies for the gradual and sudden “once-off” background changes are proposed to adapt to various changes in background through the video. The convergence of the learning process is proved and a formula to select a proper learning rate is also derived. Experiments have shown promising results in extracting foreground objects from many complex backgrounds including wavering
tree branches, flickering screens and water surfaces, moving escalators, opening and closing doors, switching lights and shadows of moving objects.

#### 2.8 码本 (CodeBook)

> 编码本的基本思路是这样的：针对每个像素在时间轴上的变动，建立多个（或者一个）包容近期所有变化的Box（变动范围）；在检测时，用当前像素与Box去比较，如果当前像素落在任何Box的范围内，则为背景。

+ K Kim, **Real-time foreground–background segmentation using codebook model**, `2005`, cited by `1400+`

+ A Ilyas, **Real-time foreground-background segmentation using a modified codebook model**, `2008`, cited by `50+`

> *Abstract* -- We present a real-time algorithm for foreground–background segmentation. Sample background values at each pixel are quantized into codebooks which represent a compressed form of background model for a long image sequence. This allows us to capture structural background variation due to periodic-like motion over a long period of time under limited memory. The
codebook representation is efficient in memory and speed compared with other background modeling techniques. Our method can handle scenes containing moving backgrounds or illumination variations, and it achieves robust detection for different types of videos. We compared our method with other multimode modeling techniques. 
> In addition to the basic algorithm, two features improving the algorithm are presented—layered modeling/detection and adaptive
codebook updating.

**Background modeling**

The CB algorithm adopts a **quantization/clustering** technique to construct a
background model from long observation sequences. For each pixel, it builds a `codebook` consisting of one or more codewords. Samples at each pixel are clustered into the set of codewords based on `a color distortion metric` together with brightness bounds. Not all pixels have the same number of codewords. The clusters represented by codewords do not necessarily correspond
to single Gaussian or other parametric distributions. Even if the distribution at a pixel were a single normal, there could be several codewords for that pixel. The background is encoded on a `pixel-by-pixel basis`. 

**Detection** 

Detection involves testing the difference of the current image from the background model with respect to `color and brightness differences`. If an incoming pixel meets two conditions, it is classified as background — (1) the color distortion to some codeword is less than the `detection threshold`, and (2) its brightness lies within the `brightness range` of that codeword. Otherwise, it is classified as foreground.

`comment`
>  效果还可以，有多种变体，对光照敏感

#### 2.9 样本一致性背景建模算法  (SACON)

+ Hanzi Wang, **A consensus-based method for tracking: Modelling background scenario and foreground appearance**, `2007`, cited by `100+`.

> *Abstract* -- Modelling of the background ("uninteresting parts of the scene"), and of the foreground, play important roles in the tasks of visual
detection and tracking of objects. This paper presents an effective and adaptive background modelling method for detecting foreground objects in both static and dynamic scenes. The proposed method computes SAmple CONsensus (SACON) of the background samples and estimates a statistical model of the background, per pixel. SACON exploits both color and motion information to detect foreground objects. SACON can deal with complex background scenarios including nonstationary scenes (such as moving trees, rain, and fountains),
moved/inserted background objects, slowly moving foreground objects, illumination changes etc. However, it is one thing to detect objects that are not likely to be part of the background; it is another task to track those objects. Sample consensus is again utilized to model the appearance of foreground objects to facilitate tracking. This appearance model is employed to
segment and track people through occlusions. Experimental results from several video sequences validate the effectiveness of the proposed method.

`comment` 

> 基于统计知识，效果还可以

#### 2.10 自组织背景建模  (SOBS: Self-organization background subtraction)

+ Lucia Maddalena, **A self-Organizing approach to background subtraction for visual surveillance Applications**, `2008`, cited by `580+`

> *Abstract* — Detection of moving objects in video streams is the first relevant step of information extraction in many computer vision applications. Aside from the intrinsic usefulness of being able to segment video streams into moving and background components, detecting moving objects provides a focus of attention for recognition, classification, and activity analysis, making these later steps more efficient. We propose an approach based on self organization through artificial neural networks, widely applied in human image processing systems and more generally in cognitive science. The proposed approach can handle scenes containing moving backgrounds, gradual illumination variations and camouflage, has no bootstrapping limitations, can include
into the background model shadows cast by moving objects, and achieves robust detection for different types of videos taken with stationary cameras. We compare our method with other modeling techniques and report experimental results, both in terms of detection accuracy and in terms of processing speed, for color video sequences that represent typical situations critical for video
surveillance systems.

`comment` 
>  对光照有一定的鲁棒性，但MAP的模型比输入图片大，计算量比较大，但是可以通过并行处理来解决算法的速度问题，可以进行尝试

#### 2.11 ViBe (A Universal Background Subtraction): 
       
+ Olivier Barnich, **ViBe: A universal background subtraction algorithm for video sequences**, `2011`, cited by `800+`

> *Abstract* — This paper presents a technique for motion detection
that incorporates several innovative mechanisms. For example,
our proposed technique stores, for each pixel, a set of values taken
in the past at the same location or in the neighborhood. It then
compares this set to the current pixel value in order to determine
whether that pixel belongs to the background, and adapts the
model by choosing randomly which values to substitute from the
background model. This approach differs from those based on
the classical belief that the oldest values should be replaced first.
Finally, when the pixel is found to be part of the background, its
value is propagated into the background model of a neighboring
pixel. We describe our method in full details (including pseudocode and the parameter values used) and compare it to other
background subtraction techniques. Efficiency figures show that
our method outperforms recent and proven state-of-the-art
methods in terms of both computation speed and detection
rate. We also analyze the performance of a downscaled version
of our algorithm to the absolute minimum of one comparison
and one byte of memory per pixel. It appears that even such
a simplified version of our algorithm performs better than
mainstream techniques. An implementation of ViBe is available
at http://www.motiondetection.org.

`comment`

> VIBE算法是Barnich的一个大作，已申请了`专利`。 
> 
> ViBe是一种像素级视频背景建模或前景检测的算法。
> 
> 利用视频第一帧图像就能完成背景建模初始化工作，根据邻近像素点之间具有相似性完成初始化和更新，依据当前图像的像素和背景模型中对应像素之间的相似性程度来检测前景目标。

**步骤：**视频帧序列 -> 第1帧 -> 初始化背景模型 -> 第2,...,N帧 -> 前景目标检测->更新背景模型

**模型初始化**： 为图像中每个像素建立一个大小为N的背景样本集，这个样本存储了该像素点邻近像素点的像素值以及过去这一点的像素值。

**像素分类/运动检测**：判断像素是前景还是背景像素。当前帧与背景样本集比较，得到相似度，根据阈值判定。

**背景模型实时更新**： 当前像素点被检测为背景像素，将按照一定概率用该像素点去更新自己的背景样本集或者是它的邻居点背景样本。

**算法的主要优势：**
- 内存占用少，没有浮点运算，计算量低，算法效率高，
- 一个像素需要作一次比较，占用一个字节的内存；
- 像素级算法，视频处理中的预处理关键步骤；
- 背景模型初始化速度极快，适用于手持相机等复杂的视频环境；
- 总体性能优于帧差发，光流法，混合高斯，SACON等，具有较好的抗噪能力。
- 可直接应用在产品中，软硬件兼容性好；

**算法的缺点：**
* 容易引入**ghost** 区域，“鬼影”。
* 对 **光照** 强弱变化等动态场景敏感，不适用动态背景下的目标检测。
* 无法消除运动目标的 **阴影** 。

**改进：**
* 自适应阈值
* 形态学处理
* 结合三帧差分、边缘检测等技术
 
`More`

+ M Van Droogenbroeck, **Background subtraction: Experiments and improvements for ViBe**, `2012`, cited by `140+`
+ 余烨, **EVibe:一种改进的Vibe运动目标检测算法**,仪器仪表学报,2014, cited by `27`

> 此算法扩大了样本的取值范围，避免了样本的重复选取;采用隔行更新方式对邻域进行更新， 避免了错误分类的扩散;采用小目标丢弃和空洞填充策略去除了噪声的影响;添加了阴影去除模块, 增强了算法对阴影的鲁棒性

+ 胡小冉, **一种新的基于ViBe的运动目标检测方法**,计算机科学,`2014`, cited by `20`

> 预处理阶段通过三帧差分获得真实背景并消除鬼影，运动目标检测阶段结合先验知识和边缘检测方法获得真实的运动目标以消除阴影，目标描述与跟踪阶段运用像素标记分割方法得到目标描述 并实现目标跟踪。


+ 桂斌, **基于ViBe的运动目标检测与阴影消除方法研究**, 安徽大学硕士论文, `2015`, cited by `0`
+ 王彬, **基于改进的ViBE和HOG的运动目标检测系统研究与实现**, 沈阳工业大学硕士论文,  `2016`, cited by `0` 

#### 2.12 Summary
SOBS、Color、VIBE、SACON、W4等可以进行深入的了解，特别是近年来出现的Block-based或Region-Based、Features-Based、基于层次分类或层次训练器的算法可以进行深入的研究。


### 3. 运动分割（motion segmentation）

In **motion segmentation**, the moving objects are continuously present in the scene, and the background may also move due to camera motion. The target is **to separate different motions**.

#### 3.1 光流法 (optical flow) 

光流是一种可以观察到的目标的运行信息。当运动目标和摄像头发生相对运动，运动目标表明所携带的光学特征就能为我们带来目标的运动信息。光流就是运动目标在成像平面上像素点运动的随机速度。是非常`经典（古老）`基于运动的目标检测方法。

> Optical flow or optic flow is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene.[1][2] The concept of optical flow was introduced by the American psychologist James J. Gibson in the 1940s to describe the visual stimulus provided to animals moving through the world.[3] Gibson stressed the importance of optic flow for affordance perception, the ability to discern possibilities for action within the environment. Followers of Gibson and his ecological approach to psychology have further demonstrated the role of the optical flow stimulus for the perception of movement by the observer in the world; perception of the shape, distance and movement of objects in the world; and the control of locomotion.[4]
> 
> The term optical flow is also used by roboticists, encompassing related techniques from image processing and control of navigation including motion detection, object segmentation, time-to-contact information, focus of expansion calculations, luminance, motion compensated encoding, and stereo disparity measurement.[5][6]
> 
>  from [wikipedia/Optical_flow](https://en.wikipedia.org/wiki/Optical_flow)
>


The dense optical flow is often used for **Motion Segmentation(运动分割)**.


+ David J. Fleet, **Optical Flow Estimation**, chapter15, `2005`, cited by `200+`
+ Stefan Roth, **On the spatial statistics of optical flow**, `2005`, cited by `260+`
+ 董颖, **基于光流场的视频运动检测**, 山东大学硕士论文, `2008`, cited by `58`.
+ 裴巧娜，**基于光流法的运动目标检测与跟踪技术**，北方工业大学硕士论文，2009, cited by `107`.

MathWorks: [Live Motion Detection Using Optical Flow](https://cn.mathworks.com/help/imaq/examples/live-motion-detection-using-optical-flow.html)


#### 3.2  运动竞争 (Motion Competition) 

+ Daniel Cremers, **Motion Competition: A Variational Approach to Piecewise Parametric Motion Segmentation**,`2005`, cited by `260+`

> *Abstract* -  We present a novel variational approach for segmenting the image plane into a set of regions of parametric motion on the basis of two consecutive frames from an image sequence. Our model is based on a conditional probability for the spatio-temporal image gradient, given a particular velocity model, and on a geometric prior on the estimated motion field favoring motion boundaries of minimal length. Exploiting the Bayesian framework, we derive a cost functional which depends on parametric motion models for each of a set of regions and on the boundary separating these regions. The resulting functional can be interpreted as an extension of the Mumford-Shah functional from intensity segmentation to motion segmentation. In contrast to most alternative approaches, the problems of segmentation and motion estimation are jointly solved by continuous minimization of a single functional. Minimizing this functional with respect to its dynamic variables results in an eigenvalue problem for the motion parameters and in a gradient descent evolution for the motion discontinuity set. We propose two different representations of this motion boundary: an explicit spline-based implementation which can be applied to the motion-based tracking of a single moving object, and an implicit multiphase level set implementation which allows for the segmentation of an arbitrary number of multiply connected moving objects. Numerical results both for simulated ground truth experiments and for real-world sequences demonstrate the capacity of our approach to segment objects based exclusively on their relative motion.

#### 3.3 DECOLOR

DEtecting Contiguous Outliers in the LOw-rank Representation (DECOLOR)

+ Xiaowei zhou, **Moving object detection by detecting contiguous outliers in the low-rank representation**, `2013`, cited by `200+`

> *Abstract* — Object detection is a fundamental step for automated video analysis in many vision applications. Object detection in a video
is usually performed by object detectors or background subtraction techniques. Often, an object detector requires manually labeled
examples to train a binary classifier, while background subtraction needs a training sequence that contains no objects to build a
background model. To automate the analysis, object detection without a separate training phase becomes a critical task. People have
tried to tackle this task by using motion information. But existing motion-based methods are usually limited when coping with complex
scenarios such as nonrigid motion and dynamic background. In this paper, we show that the above challenges can be addressed in a
unified framework named DEtecting Contiguous Outliers in the LOw-rank Representation (DECOLOR). This formulation integrates
object detection and background learning into a single process of optimization, which can be solved by an alternating algorithm
efficiently. We explain the relations between DECOLOR and other sparsity-based methods. Experiments on both simulated data and
real sequences demonstrate that DECOLOR outperforms the state-of-the-art approaches and it can work effectively on a wide range of
complex scenarios.

> *Index Terms* — Moving object detection, low-rank modeling, Markov Random Fields, motion segmentation


#### 3.4 Long Term Video Analysis

+  Peter Ochs, **Segmentation of Moving Objects by Long Term Video Analysis**, `2014`, cited by `130+`

> *Abstract* — Motion is a strong cue for unsupervised object-level grouping. In this paper, we demonstrate that motion will be exploited
most effectively, if it is regarded over larger time windows. Opposed to classical two-frame optical flow, point trajectories that span
hundreds of frames are less susceptible to short-term variations that hinder separating different objects. As a positive side effect, the
resulting groupings are temporally consistent over a whole video shot, a property that requires tedious post-processing in the vast
majority of existing approaches. We suggest working with a paradigm that starts with semi-dense motion cues first and that fills up
textureless areas afterwards based on color. This paper also contributes the Freiburg-Berkeley motion segmentation (FBMS) dataset,
a large, heterogeneous benchmark with 59 sequences and pixel-accurate ground truth annotation of moving objects.

> *Index Terms* — Motion segmentation, point trajectories, variational methods

### 4. 其他方法

#### 4.1 运动历史图像 （motion history image, MHI）

+ James W. Davis, **Hierarchical Motion History Images for Recognizing Human Motion**, `2001`, cited by `170+`
+ MAR Ahad, **Motion history image: its variants and applications**, `2012`, cited by `170+`

> The motion history image (MHI) is a static image template helps in understanding the motion location and path as it progresses.[1] In MHI, the temporal motion information is collapsed into a single image template where intensity is a function of recency of motion. Thus, the MHI pixel intensity is a function of the motion history at that location, where brighter values correspond to a more recent motion. Using MHI, moving parts of a video sequence can be engraved with a single image, from where one can predict the motion flow as well as the moving parts of the video action.

> 
> Some important features of the MHI representation are: 
> 
>   - It represents motion sequence in a compact manner. In this case, the silhouette sequence is condensed into a grayscale image, where dominant motion information is preserved.
>   - MHI can be created and implemented in low illumination conditions where the structure cannot be easily detected otherwise.
>   - The MHI representation is not so sensitive to silhouette noises, holes, shadows, and missing parts.
>   - The gray-scale MHI is sensitive to the direction of motion because it can demonstrate the flow direction of the motion.
>   - It keeps a history of temporal changes at each pixel location, which then decays over time.
>   - The MHI expresses the motion flow or sequence by using the intensity of every pixel in a temporal manner.
> 
>  from [wikipedia/Motion_History_Images](https://en.wikipedia.org/wiki/Motion_History_Images)


## Recent papers

1. Pierre-Luc St-Charles, **SuBSENSE: A Universal Change Detection Method With Local Adaptive Sensitivity**, `2015`, cited by `80+`

## Survey
1. W Hu, **A survey on visual surveillance of object motion and behaviors**, `2004`, cited by `2300+`
2. M Piccardi, **Background subtraction techniques: A review**, `2004`, cited by `1900+`
2. Thomas B. Moeslund, **A survey of advances in vision-based human motion capture and analysis**, `2006`, cited by `2400+`
3. S Brutzer, **Evaluation of Background Subtraction Techniques for Video Surveillance**, `2011`, cited by `400+`
4. A Sobral, **A comprehensive review of background subtraction algorithms evaluated with synthetic and real videos**, `2014`, cited by `200+`
5. T Bouwmans, **Traditional and recent approaches in background modeling for foreground detection  An overview**, `2014`, cited by `180+` 

## Library/Software

**Background subtraction Library**
1. [**BGSLibrary**](https://github.com/andrewssobral/bgslibrary): The BGS Library (A. Sobral, Univ. La Rochelle, France) provides a C++ framework to perform background subtraction algorithms. The code works either on Windows or on Linux. Currently the library offers more than 30 BGS algorithms. 
2. [**LRS Library**](https://github.com/andrewssobral/lrslibrary) - Low-Rank and Sparse tools for Background Modeling and Subtraction in Videos. The LRSLibrary (A. Sobral, Univ. La Rochelle, France) provides a collection of low-rank and sparse decomposition algorithms in MATLAB. The library was designed for motion segmentation in videos, but it can be also used or adapted for other computer vision problems. Currently the LRSLibrary contains more than 100 matrix-based and tensor-based algorithms. 

**Other libary**
1. [**Motion Detection Algorithms**](https://www.codeproject.com/Articles/10248/Motion-Detection-Algorithms): There are many approaches for motion detection in a continuous video stream. All of them are based on comparing of the current video frame with one from the previous frames or with something that we'll call background. In this article, I'll try to describe some of the most common approaches.

## Reference 

1. 目标检测中背景建模方法 [http://www.cnblogs.com/ronny/archive/2012/04/12/2444053.html](http://www.cnblogs.com/ronny/archive/2012/04/12/2444053.html)
2. A video database for testing change detection algorithms [http://www.changedetection.net/](http://www.changedetection.net/)
