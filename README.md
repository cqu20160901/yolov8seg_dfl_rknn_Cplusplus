# yolov8seg_dfl_rknn_Cplusplus
yolov8seg 瑞芯微 rknn 板端 C++部署，使用平台 rk3588，部署难度小，运行速度快。

# yolov8seg_rknn_Cplusplus

yolov8seg 瑞芯微 rknn 板端 C++部署，使用平台 rk3588。

## 编译和运行

1）编译

```
cd examples/rknn_yolov8seg_demo_open

bash build-linux_RK3588.sh

```

2）运行

```
cd install/rknn_yolov8seg_demo_Linux

./rknn_yolov8seg_demo

```

注意：修改模型、测试图像、保存图像的路径，修改文件为src下的main.cc

```

int main(int argc, char **argv)
{
    char model_path[256] = "/home/zhangqian/rknn/examples/rknn_yolov8seg_demo_dfl_open/model/RK3588/yolov8nseg_relu_80class_dfl.rknn";
    char image_path[256] = "/home/zhangqian/rknn/examples/rknn_yolov8seg_demo_dfl_open/test.jpg";
    char save_image_path[256] = "/home/zhangqian/rknn/examples/rknn_yolov8seg_demo_dfl_open/test_result.jpg";

    detect(model_path, image_path, save_image_path);
    return 0;
}
```


# 板端测试效果

冒号“:”前的数子是coco的80类对应的类别，后面的浮点数是目标得分。（类别:得分）

pytorch 效果

![image](https://github.com/cqu20160901/yolov8seg_rknn_Cplusplus/assets/22290931/a771924b-8725-444b-81e5-e7e36d37722d)

板端推理效果

![images](https://github.com/cqu20160901/yolov8seg_dfl_rknn_Cplusplus/blob/main/examples/rknn_yolov8seg_demo_dfl_open/test_result.jpg)


说明：推理测试预处理没有考虑等比率缩放，训练时激活函数 SiLU 用 Relu 进行了替换。由于使用的数据不多，效果并不是很好，仅供测试流程用。

把板端模型推理和后处理时耗也附上，供参考，使用的芯片rk3588，输入分辨率640x640。
![image](https://github.com/cqu20160901/yolov8seg_dfl_rknn_Cplusplus/assets/22290931/8292935c-c7ec-4192-876e-64fc52210575)


