## 官方手册

[Project Jupyter 文档 — Jupyter 文档 4.1.1 Alpha 文档 --- Project Jupyter Documentation — Jupyter Documentation 4.1.1 alpha documentation](https://docs.jupyter.org/en/latest/)

jupyter就是一个python编译器，只不过套了一层浏览器的外衣

## 启动Jupyter notebook

1. 在终端输入回车

```
jupyter notebook
```

2. 十几秒内，会自动打开一个网页（此终端不要关闭）
3. 如果你在桌面启动的终端，此时网页上的目录就是桌面的目录
4. 如何在想要的地方启动终端？可以在该文件夹的搜索框内输入cmd![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601090842.XRlL5uhh.png)
5. 在这个终端页面输入jupyter notebook,此时会以此目录打开网页
6. 另外建议将Chrome设为默认浏览器

## 界面介绍

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601091831.BVmUAX_B.png)在浏览器页面 选择右侧的NEW，新建一个Notebook文件

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601091602.CcZdMWIz.png)

在光标闪烁的位置输入代码 按下

> ctrl enter

就会执行代码![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601091852.D3uerdG-.png)

### 笔记的单元格

一个cell就是一个单元格

有两种状态：

1. 编辑状态（鼠标键入，有光标闪烁时）
2. 选中状态（点击cell输入框旁边的空白区域）

编辑状态按ESC快速选中状态 选中状态按Enter快速进入编辑状态

#### 对单元格的操作（都在选中状态下操作）

1. 增加单元格（单击b键在下方新增，单击a在上方新增）
2. 删除单元格（双击d，删除当前选中的单元格）
3. 剪切单元格（单击x就剪切了，选中一个单元格位置，再单击v，会将剪切的粘到当前单元格的下方）
4. 撤销操作（任意位置单击z键）

有两种以上的模式![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601094420.D898xh3O.png)

1. Code模式就是python能运行的代码
2. Markdown跟做笔记一样，主要是写一些说明（写完之后记得按Ctrl enter，文本才会进入预览模式）![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601094941.CnV8TdRu.png)

不会Markdown的可以看这个网站，5分钟学会，能用一辈子 [基本语法 | MARKDOWN 中文](https://www.markdown.cn/docs/tutorial-basics/basic-syntax)

选中时按y进入code模式 按m进入markdown模式

想给当前这个notebook改名的话可以这样操作![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601095306.DDUN40Xo.png)