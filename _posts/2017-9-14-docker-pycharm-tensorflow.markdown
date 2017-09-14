---
layout:     post
title:      "docker+pycharm+tensorflow开发深度学习项目"
subtitle:   " "
date:       2017-09-14
author:     "板锅锅"
header-img: "img/post-bg-universe.jpg"
catalog:    true
tags:
    - docker
    - pycharm
    - tensorflow
    - deep learning
---

---

## 前言

在经历了无数次被安装开发环境导致的各种匪夷所思的错误折磨的一周啥也做不成的惨痛教训之后，我们终于决定用Docker容器来配置我们的日常开发环境。那么问题来了，Docker的好处都有啥？查阅官方文档有真相。对于我们而言最大的好处就是1. 简化开发环境的安装和配置；2. 环境隔离，不用担心不同底层依赖的冲突，如gcc版本，opencv版本等等。如此一来，算法同学们可以把99%的精力放到算法研发上来，远离环境安装和配置的苦恼。
针对上面的问题，只需要在机器上配置好Docker环境，习惯于命令行和上古文本编辑器（vim，emacs）开发的大神们已经可以发车了。但是，对于像作者这种依赖IDE的菜鸡，还是希望引入Docker后不会影响日常的开发需求，具体包括：

1. 在Docekr环境下能像本地一样流畅地进行项目迭代，编辑->测试->运行一种循环。
2. 能够在Docker环境下正常跑实验，即读取本地文件系统的数据，再将训练出的模型输出到本地。
3. 在Docker容器内继续使用IDE的常用功能如代码补全、代码跳转、debug和profiling。

“世上本没有路，走的人多了，便成了路”。幸运的是，开源社区的老司机们早已为我们趟出了一条路。

1和2用Docker的命令行工具加上一长串参数就可以满足，[docker-compose](https://docs.docker.com/compose/)的出现极大地方便了基于Docker的开发和部署。为了让Docker容器能够正常使用GPU，NVIDIA自己推出了[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)，有人在此基础上开发了[nvidia-docker-compose](https://github.com/eywalker/nvidia-docker-compose)。它们的命令行参数和原有的官方版本基本一致。

3则需要IDE本身提供Docker的集成。目前主流的深度学习框架都提供良好的python API，所以用python IDE来开发深度学习相关的项目成为了不二选择。而[PyCharm](https://blog.jetbrains.com/pycharm/2017/03/docker-compose-getting-flask-up-and-running/)在2017 Professional版本里集成了Docker和Docker Compose。如此一来，一套完整的基于Docker+IDE的日常开发环境已经呼之欲出了。下面就以基于[tensorflow](https://www.tensorflow.org/)的demo为例，来演示如何用PyCharm完成日常开发和调试，并在Docker容器内用MNIST数据在GPU上训练一个手写数字的CNN分类模型，所有代码及配置文件在[这里](https://github.com/fanOfJava/tf-docker-pycharm-demo)。只要按下面的教程配置成功，就可以在自己的机器上通过Docker运行示例代码。

---

## 准备工作

该教程在Ubuntu 16.04以及 Debian 3.16.36-1+deb8u2 (2016-10-19)上测试成功，主流的Linux发行版应该都可以。
需要在机器上安装docker，docker-compose，nvidia-docker，nvidia-docker-compose
需要安装PyCharm Professional 2017.1.4及以上版本。至于license么，懂得自然懂。
阅读docker相关的文档，了解基本概念，熟悉常用命令。（注：docker官方文档非常清晰详实）

---

## 安装配置Docker

1. 按官方教程docker ce（community edtion），并按教程测试docker是否成功安装。
2. (可选）Linux环境下的权限变更，这样不用每次docker命令都加sudo，自己没服务器权限找管理员变更。
3. 按教程安装nvidia-docker。
4. 按官方教程安装docker-compose，并按教程测试docker是否成功安装。
5. 按教程安装nvidia-docker-compose

==注意==：3如果不用GPU不是必须的。4和5也不是必须的，但是为了在PyCharm中集成最好还是装上

---

## 在Docker中使用Tensorflow

不需要或者无法使用GUI环境的时候，用命令行就是可以直接在Docker容器内跑tensorflow了，tensorflow官网给出了官方维护的Docker镜像地址。
如果成功安装了docker及nvidia-docker，在终端中运行

``` shell
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu
```
就可以启动一个基于tensorflow/tensorflow:latest-gpu的docker container，命令中的参数含义可以参阅docker文档，这里的-it相当于是打开交互命令行窗口，-p则将container内的8888端口映射到host上的8888端口。tensorflow的这个镜像默认会启动jupyter notebook，根据命令行提示信息在本机浏览器中输入url地址如：

``` shell
http://localhost:8888/?token=1af80c20f70246419099ecd6528d800147924a1c702def69
```
就可以打开运行在docker容器内的notebook了


![img](/img/posts/docker-pycharm-tensorflow/0.png)

---

## 在Pycharm上集成Docker

为了演示如何在pycharm上集成docker来完成日常开发，将示例工程拉到自己本地并在PyCharm中打开，除了python代码外还有几个docker环境需要的配置文件。

Dockerfile配置文件定义如何构建该工程的docker镜像。

``` shell
#基于tensorflow/tensorflow:latest-gpu的docker镜像构建
FROM tensorflow/tensorflow:latest-gpu
#暴露8888端口
EXPOSE 8888
#执行mkdir命令
RUN mkdir /app
#启动container后的默认路径为/app
WORKDIR /app
#将requirements.txt文件复制到镜像中的/app/requirements.txt
COPY requirements.txt /app/requirements.txt
#执行pip命令
RUN pip install -r requirements.txt
#将当前目录复制到镜像/app目录下
COPY . /app
```
docker-compose.yml配置文件定义启动相关服务（service）的docker container。

``` shell
# yml配置文件格式版本为2
version: '2'
services:
# 定义一个名叫web的service
  web:
# 基于当前目录构建镜像
    build: .
# 镜像名为tf-demo/docker:0.0.1
    image: tf-demo/docker:0.0.1
# 映射端口
    ports:
    - "8888:8888"
    - "8887:6006"
# 映射文件路径
    volumes:
    - .:/app
```
docker和docker-compose的配置文件定义好后，我们利用nvidia-docker-compose来自动生成一个GPU版本的yml配置文件，在工程根目录下执行

``` shell
nvidia-docker-compose up
```
执行成功后会在根目录下生成一个nvidia-docker-compose.yml的配置文件，观察文件内容可以发现nvidia-docker-compose在配置文件里加入了GPU相关的内容，这个在配置PyCharm的时候要用到。

打开PyCharm，在File->Settings->Build,Execution,Deployment→Docker中配置Docker（进入该界面时应该会自动生成一些配置）

![img](/img/posts/docker-pycharm-tensorflow/1.png)

注意Certificates folder一栏留空,然后在Project→Project Interpreter中配置

![img](/img/posts/docker-pycharm-tensorflow/2.png)

点击右侧小齿轮，选择add remote后

![img](/img/posts/docker-pycharm-tensorflow/3.png)

选择Docker Compose，剩下的一般自动生成，这里注意将Configuration file改为nvidia-docker-compose，这样通过PyCharm启动的docker container可以使用GPU。
这个配置好后，在PyCharm编辑代码时，所有的代码补全和跳转都是基于docker image中的库，所以即便本地没有安装tensorflow，你也可以正常import。
接下来创建一个可执行程序

![img](/img/posts/docker-pycharm-tensorflow/4.png)

点击右上角的向下箭头选择Edit Configuration

![img](/img/posts/docker-pycharm-tensorflow/5.png)

随后选择要执行的脚本，这里注意Python Interpreter是刚才配置好的remote python。配置好后，不管是运行还是调试模式，PyCharm都会启动docker container在其中运行。

---
## 结语
”纸上得来终觉浅，绝知此事要躬行“。还犹豫什么，自己动手试试吧








