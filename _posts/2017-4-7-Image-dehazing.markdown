---
layout:     post
title:    "除雾算法最新进展"
subtitle:   "Recent developments in Image Haze Removal"
author:     "huajh7"
catalog:    true
header-img: "img/post-bg-universe.jpg"
tags:
  - Computer Vision
  - Image Processing
  - Image Haze Removal
date: 2017-4-7  20:02:02
--- 

除雾算法主要包括 1) 图像增强方法，和2) 基于物理模型的方法。

后者又包括(1)基于景物深度信息, (2)基于大气光偏振特性, 以及(3)基于雾天图像的先验知识。

下面总结下基于雾天图像的先验知识的去雾算法。

#### Maximum Contrast 
 
> 基于统计信息, 认为无雾图像相对于有雾图像来说对比度要高得多
> 
> 根据大气散射模型，雾霾会降低物体成像的对比度. 因此，基于这个推论可利用局部对比度来近似估计雾霾的浓度。同时，也可以通过最大化局部对比度来还原图像的颜色和能见度。

RT Tan, **Visibility in bad weather from a single image**, CVPR, `2008`, cited by `1000+`


#### Fattal

>这种方法是基于物理的复原模型, 复原图像自然且能求出良好的深度图. 然而, 这种方法是基于彩色图像的统计特性的,因而该方法也无法作用于灰度图像, 而且这个统计特性在浓雾区域和低信噪比区域会失效

Raanan Fattal, **Single image dehazing**, TOG, `2008`, cited by `1100+`


#### Dark Channel Prior

> 说起去雾特征，不得不提起的暗通道先验（DCP）。大道之行在于简，DCP作为CVPR 2009的最佳论文，以简洁有效的先验假设解决了雾霾浓度估计问题。

> 观察发现，清晰图像块的RGB颜色空间中有一个通道很暗（数值很低甚至接近于零）。因此基于暗通道先验，雾的浓度可由最暗通道的数值近似表示.

Kaiming He, **Single Image Haze Removal Using Dark Channel Prior**, CVPR/PAMI, `2009/2011`, cited by `1800+`.

> 该方法具有革命性, 简单有效, 去雾效果理想, 处理后图像颜色自然逼真, 少有地用一个简单得不可思议的方法使一个复杂问题的实验效果得到巨大的提升.

* propose the **Dark Channel Prior**
* soft matting过程比较复杂，并且执行速度非常慢

Kaiming He, **Guided Image Filtering**, ECCV/PAMI, `2010/2013`, cited by `1990+` 

> 导向滤波来代替soft matting的过程，且速度很快
> 
> 暗通道先验去雾算法的参数需要根据不同的图像手动地作出调整, 无法自适应调整.
> 该方法所使用的软抠图算法需要进行大型稀疏矩阵的运算,时间和空间复杂度都极高,无法实时处理大幅图片, 而且当景物颜色与天空颜色接近且没有阴影时, 暗原色先验失效, 该算法也随之失效。
> 
> 后来该文献的作者 He 又使用了引导滤波替代软抠图处理, 较大地提高了效率 (600像素 x 400 像素图像处理时间从 10 秒变为 0.1 秒)的同时, 去雾效果基本不变

...

有很多改进算法

...

Zhengguo Li, **Weighted Guided Image Filtering**, `2015`, cited by `40+`


#### Tarel

Tarel 假设大气耗散函数 (Atmosphericveil) 在局部上变化平缓, 因此用中值滤波代替 He等的算法中的最小值滤波来对介质透射系数进行估计.

Jean-Philippe Tarel, **Fast visibility restoration from a single color or gray level image**, CVPR, `2009`, cited by `600+`

> He与Tarel的方法简单有效,尤其He提出的暗原色先验去雾算法是图像去雾领域的一个`重要突破`, 为图像去雾的研究人员提供了一个新思路,后来出现的许多去雾算法都是基于这两种算法的改进或补充
> 

#### Color  Attenuation Prior 
> 
> 作者提出了一个简单，但是很有效的先验：颜色衰减先验（CAP），用来通过仅仅输入一张有雾的图像来去除雾的影响。这是一种与暗通道先验（DCP）相似的先验特征。
> 
> 作者观察发现雾霾会同时导致图像饱和度的降低和亮度的增加，整体上表现为颜色的衰减。根据颜色衰减先验，亮度和饱和度的差值被应用于估计雾霾的浓度.
> 
> 作者创建了一个线性回归模型，利用颜色衰减先验这个新奇的先验，通过对有雾图像场景深度的建模，利用有监督学习的方法学习到的参数，深度信息会被很好的恢复。利用有雾图像的深度图，我们可以很容易的恢复一张有雾的图像。
> 

Qingsong Zhu, **A fast single image haze removal algorithm using color attenuation prior**, TIP, `2015`, cited by `60+`

Project page: [https://github.com/JiamingMai/Color-Attenuation-Prior-Dehazing](https://github.com/JiamingMai/Color-Attenuation-Prior-Dehazing)

`expermental results`
![img](/img/posts/haze-removal/post-haze-removal-zhu2016.jpg)

## 3. 综述

吴迪, **图像去雾的最新研究进展**, 自动化学报, `2015`, cited by `55`.

## 4. 最新文献


`deep learning`
> DehazeNet是一个特殊设计的深度卷积网络，利用深度学习去智能地学习雾霾特征，解决手工特征设计的难点和痛点。
> 

Bolun Cai, **DehazeNet: An End-to-End System for Single Image Haze Removal**, `2016`, cited by `9`

* Project page: [http://caibolun.github.io/DehazeNet/](http://caibolun.github.io/DehazeNet/)
* Code: [https://github.com/caibolun/DehazeNet](https://github.com/caibolun/DehazeNet)

Dana Berman, **Non-Local Image Dehazing**, CVPR, 2016, cited by `7`

Mostafa M. El-Hashash, **High-speed video haze removal algorithm for embedded systems**, `2016`, cited by `0`

* Real-time video processing
* uses the dark channel prior 
* eight frames per second at 720 x 480 video frame resolution


Adobe, Photoshop Lightroom CC, [http://www.adobe.com/products/photoshop-lightroom/features.html](http://www.adobe.com/products/photoshop-lightroom/features.html)




