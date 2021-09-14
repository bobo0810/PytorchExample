# 标签格式转化

- [x] YOLO ->VOC



## YOLO格式

```shell
images
    test.jpg
    ...
labels
    test.txt
    ...
```

test.txt内容

```shell
# 类别序号 锚框中心点x  锚框中心点y  锚框宽 锚框高
# 坐标经过归一化
2 0.61 0.64 0.12 0.20
```



## VOC格式

```
JPEGImages
		test.jpg
		...
Annotations
		test.xml
		...
```

test.xml内容

```xml
<annotation>
    <folder>VOC2007</folder>
    <filename>test.jpg</filename>
    <size>							<! --图像尺寸-->
        <width>720</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>person</name>  		<! --锚框类别名称-->
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>    				<! --锚框左上角、右下角坐标-->
            <xmin>395</xmin>
            <ymin>261</ymin>
            <xmax>486</xmax>
            <ymax>360</ymax>
        </bndbox>
    </object>
</annotation>
```

