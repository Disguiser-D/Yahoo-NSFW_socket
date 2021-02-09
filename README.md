# 基于雅虎NSFW开源模型的socket程序

这个版本库包含了 [Yahoo's Open NSFW Classifier](https://github.com/yahoo/open_nsfw) 在tensorflow中重写的一个实现，以及[tensorflow-open_nsfw](https://github.com/mdietrichstein/tensorflow-open_nsfw) 中的一些Tools实现

原始的caffe权重已经用 [Caffe to TensorFlow](https://github.com/ethereon/caffe-tensorflow) 提取出来了你可以在`data/open_nsfw-weights.npy`中找到它们。你可以在`data/open_nsfw-weights.npy`找到它们。

## 运行环境

所有的代码已在 `Python 3.6` 和 `Tensorflow 1.x` (tested with 1.12)  和`Numpy1.16.X` 和`Skimage 0.15.X`  上成功运行。不建议使用与推荐环境相差过大的版本，否则可能会出现无法运行的情况或其他异常。模型的实现可以在`model.py`中找到。

### 使用

在使用前应当根据自己的情况更改config.ini

```
> python classify_nsfw.py
```

输入命令后classify_nsfw.py将创建一个socket服务端，发送图片到自己指定的端口，将会返回一个图片的识别情况

__提示：__目前仅支持Jepg，虽然PNG也能进行使用但对此结果并不做任何保证。



`classify_nsfw.py` 使用同一目录下的config.ini读取相关配置:

__BUFSIZ__

设置socket接收图片单次的buffer size大小

__HOST__

设置socket服务器端运行的host(IP)

__PORT__

设置socket服务器端运行的端口

__PROCESS_NUMBER__

设置进程池中的进程数，如果设置的数字小于或等于0，那么程序将自动根据系统的核心数来设置进程数

__FN_LOAD_IMAGE__

分类工具支持两种不同的图像加载机制。

* `yahoo` 复制yahoo的原始图像加载和预处理。如果你想得到与原始实现相同的结果，请使用这个选项。
* `tensorflow` 是一个完全使用tensorflow的图像加载器（不依赖`PIL`、`skimage`等）。试图复制原始caffe实现中使用的图像加载机制，虽然由于jpeg和调整大小的实现不同，但还是有一点不同。详见[本期](https://github.com/mdietrichstein/tensorflow-open_nsfw/issues/2#issuecomment-346125345)。

__注意：__ 根据所选择的图像加载器，分类结果可能会有所不同！

__INPUT_TYPE__

确定模型内部是否使用浮动张量(`tensor` - `[None, 224, 224, 3]` - 默认)或base64编码的字符串张量(`base64_jpeg` - `[None, ]`)作为输入。如果使用`base64_jpeg`，那么将使用`tensorflow`图像加载器，如果你使用了`base64_jpeg`的输入模式而使用`yahoo`图片加载器，那么程序将会提示你参数不兼容(parameter mismatch)

__LEVEL__

输出日志的级别

__FILENAME__

输出日志文件的文件名

### 工具

`tools`文件夹中包含了一些用于测试模型的实用脚本。

__create_predict_request.py__。

获取一个输入图像，并生成一个适合预测请求的 json 文件，以提交给用 [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/prediction-overview) (`gcloud ml-engine predict`) 或 [tensorflow-serving](https://www.tensorflow.org/serving/) 部署的 Open NSFW 模型。


_export_savedmodel.py__。

使用 tensorflow serving export api (`SavedModel`)导出模型。该导出可用于将模型部署在 [Google Cloud ML Engine](https://cloud.google.com/ml-engine/docs/concepts/prediction-overview)、[Tensorflow Serving]()或移动设备上（还没试过那个）。

__export_tflite.py__。

以[TFLite格式](https://www.tensorflow.org/lite/)导出模型。如果你想在移动或物联网设备上运行推理，请使用这个。请注意，`base64_jpeg`输入类型不能在TFLite中使用，因为标准运行时缺乏一些所需的tensorflow操作。

__export_graph.py__

输出tensorflow图和检查点。按默认情况冻结和优化图，以改善推理和部署使用（如Android、iOS等）。用`tf.import_graph_def`导入图。

