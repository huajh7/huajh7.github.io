---
layout:     post
title:    "Markdown简介"
subtitle:   "markdown"
author:     "huajh7"
catalog:    true
header-img: "img/post-bg-universe.jpg"
tags:
  - markdown  
date: 2017-4-4
---

# 欢迎使用马克飞象
 
- **功能丰富** ：支持高亮代码块、*LaTeX* 公式、流程图，本地图片以及附件上传，甚至截图粘贴，工作学习好帮手；
- **得心应手** ：简洁高效的编辑器，提供[桌面客户端][1]以及[离线Chrome App][2]，支持移动端 Web；
- **深度整合** ：支持选择笔记本和添加标签，支持从印象笔记跳转编辑，轻松管理。

-------------------

* TOC
{:toc}

## Markdown简介

> Markdown 是一种轻量级标记语言，它允许人们使用易读易写的纯文本格式编写文档，然后转换成格式丰富的HTML页面。    —— [维基百科](https://zh.wikipedia.org/wiki/Markdown)

正如您在阅读的这份文档，它使用简单的符号标识不同的标题，将某些文字标记为**粗体**或者*斜体*，创建一个[链接](http://www.example.com)或一个脚注[^demo]。下面列举了几个高级功能，更多语法请按`Ctrl + /`查看帮助。 

### 代码块

using `molokai.css`  by 
``rougify style molokai > css/syntax.css ``

Add the code in `syntax.css` to change the font and font-size

```css
.highlight {
  color: #F5F5F5;
  background-color: #272822;  /*#1b1d1e  f8f8f2  272822*/
  font-family: 'Consolas', serif;
  font-size: 15px;  
}
```


``` python
@requires_authorization
def somefunc(param1='', param2=0):
    '''A docstring'''
    if param1 > param2: # interesting
        print 'Greater'
    return (param2 - param1 + 1) or None
class SomeClass:
    pass
>>> message = '''interpreter
... prompt'''
```

`ruby`

```ruby
def show
  @widget = Widget(params[:id])
  respond_to do |format|
    format.html # show.html.erb
    format.json { render json: @widget }
  end
end
```

`matlab`

```matlab
A = cat( 3, [1 2 3; 9 8 7; 4 6 5], [0 3 2; 8 8 4; 5 3 5], ...
                 [6 4 7; 6 8 5; 5 4 3]);
% The EIG function is applied to each of the horizontal 'slices' of A.
for i = 1:3
    eig(squeeze(A(i,:,:)))
end
```

`python`

```python
class ReqStrSugRepr(type):

    def __init__(cls, name, bases, attrd):
        super(ReqStrSugRepr, cls).__init__(name, bases, attrd)

        if '__str__' not in attrd:
            raise TypeError("Class requires overriding of __str__()")

        if '__repr__' not in attrd:
            warn(
                'Class suggets overrding of __repr__()\n',
                stacklevel=3)
```


`c/c++`

```c
int main()
{
    int a = 10;
    print("int a = %d\n", a);
}
```

`java`

```java
import JavaBeans.People.Administor;

public class AcceptMember extends HttpServlet {
    /**
     * 
     * @throws ServletException if an error occurred
     * @throws IOException if an error occurred
     */
    public void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        response.setContentType("text/html");
        doPost(request, response);
    }
}


```


### LaTeX 公式

可以创建行内公式，例如 $$\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N$$。或者块级公式：

$$  x = \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$

### 表格

| First cell|Second cell|Third cell
| First | Second | Third |
First | Second | | Fourth |

 | Item | Value | Qty  |
 | :--------: | :-----:| :--: |
 | Computer  | 1600 USD |  5   |
 | Phone     |   12 USD |  12  | 
 | Pipe      |    1 USD | 234  |

### 流程图
```flow
st=>start: Start
e=>end
op=>operation: My Operation
cond=>condition: Yes or No?

st->op->cond
cond(yes)->e
cond(no)->op
```

以及时序图:

```sequence
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
```

> **提示：**想了解更多，请查看**流程图**[语法][3]以及**时序图**[语法][4]。

### 复选框

使用 `- [ ]` 和 `- [x]` 语法可以创建复选框，实现 todo-list 等功能。例如：

- [x] 已完成事项
- [ ] 待办事项1
- [ ] 待办事项2

> **注意：**目前支持尚不完全，在印象笔记中勾选复选框是无效、不能同步的，所以必须在**马克飞象**中修改 Markdown 原文才可生效。下个版本将会全面支持。


质能方程公式：$$E=mc^2$$

$$\min_{x,y} \sum_{i=1}^{N} (x_i+ 10 + y_i)^2 $$



### 脚注（footnote）

实现方式如下:

比如PHP[^1] Markdown Extra [^2] 是这样的。

[^1]: Markdown是一种纯文本标记语言

[^2]: 开源笔记平台，支持Markdown和笔记直接发为博文


### 引用方式：
I get 10 times more traffic from [Google][1] than from [Yahoo][2] or [MSN][3].  


### 下划线

---下划线---

### Images 

#### 内联方式：

![lenaNoise](/img/lenanoise.jpg "lenaNoise")


#### 引用方式：
![alt text][id] 

[id]: /img/mona-leber-final.jpg "mona-leber"


![]()



[1]: http://google.com/        "Google" 
[2]: http://search.yahoo.com/  "Yahoo Search" 
[3]: http://search.msn.com/    "MSN Search"




## 反馈与建议
- 微博：[@马克飞象](http://weibo.com/u/2788354117)，[@GGock](http://weibo.com/ggock "开发者个人账号")
- 邮箱：<hustgock@gmail.com>

---------
感谢阅读这份帮助文档。请点击右上角，绑定印象笔记账号，开启全新的记录与分享体验吧。




[^demo]: 这是一个示例脚注。请查阅 [MultiMarkdown 文档](https://github.com/fletcher/MultiMarkdown/wiki/MultiMarkdown-Syntax-Guide#footnotes) 关于脚注的说明。 **限制：** 印象笔记的笔记内容使用 [ENML][5] 格式，基于 HTML，但是不支持某些标签和属性，例如id，这就导致`脚注`和`TOC`无法正常点击。


  [1]: http://maxiang.info/client_zh
  [2]: https://chrome.google.com/webstore/detail/kidnkfckhbdkfgbicccmdggmpgogehop
  [3]: http://adrai.github.io/flowchart.js/
  [4]: http://bramp.github.io/js-sequence-diagrams/
  [5]: https://dev.yinxiang.com/doc/articles/enml.php

