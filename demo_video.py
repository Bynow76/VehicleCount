# ----------------------------------------------------------------------------------------------------------------------
# 识别视频
# 识别视频和识别图片的本质是一样的，视频可以分为一个个frame，然后进行识别
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import imutils
import cv2
import time

# ----------------------------------------------------------------------------------------------------------------------

"""
# load the COCO class labels our YOLO model was trained on
# 加载存放可以识别物体的名称，例如car,people,truck...，将得到的名称存放在LABELS中
# LABELS ---> ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
#			   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
#			   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
#			   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
#			   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
#			   'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 
#			   'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
#			   'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
#			   'toothbrush']
#
# initialize a list of colors to represent each possible class label
# 一张图片可以有上述多个可以被识别的物体，使用框圈起来的时候，要使用不一样的颜色
# np.random.seed() ---> 设置随即种子
#
# derive the paths to the YOLO weights and model configuration
# 加载权重和相应的配置数据
#
# load our YOLO object detector trained on COCO dataset (80 classes)
# 加载好数据之后，开始利用上述数据恢复yolo训练的时候神经网络
# 神经网络 ---> net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# 获取YOLO输出层的名称
# ln = net.getLayerNames() ---> 得到神经网络层的名称 ---> ['conv_0', 'bn_0', 'relu_0', 'conv_1', 'bn_1', 'relu_1', 'conv_2', 'bn_2', 'relu_2'...]
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] ---> ['yolo_82', 'yolo_94', 'yolo_106']

"""

# ----------------------------------------------------------------------------------------------------------------------

labelsPath = "./yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = "./yolo-coco/yolov3.weights"
configPath = "./yolo-coco/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ----------------------------------------------------------------------------------------------------------------------

"""
# cv2.VideoCapture() --- 里面参数如果地址，打开视频文件 --- 里面参数是0/1，打开摄像头
# 当参数是0的时候，打开计算机的内置摄像头，当参数为1的时候打开计算机的外置摄像头
# writer = None --- 设置一个标志位，当writer = None的时候将处理得到的数据写入硬盘
# (W, H) = (None, None) --- 视频的宽度和高度，初始化视频编写器（writer）和帧尺寸

"""
vs = cv2.VideoCapture("./input/cam.mp4")
writer = None
(W, H) = (None, None)

# ----------------------------------------------------------------------------------------------------------------------

"""
# try to determine the total number of frames in the video file
# 打开一个指向视频文件的文件指针，循环读取帧 --- 尝试确定视频文件中的总帧数（total），以便估计整个视频的处理时间;
# CV_CAP_PROP_FRAME_COUNT --- 视频的帧数
# 这里使用是处理视频的时候固定的过程，不必过度的纠结其使用 ---
#					   if imutils.is_cv2():
#					   	  prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
#					   else:
#					   	  prop = cv2.CAP_PROP_FRAME_COUNT
#
# vs.get(prop) --- cv2.VideoCapture.get(prop) --- 得到视频的总帧数 
# print("[INFO] {} total frames in video".format(total)) --- 输出视频的帧数

"""
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# ----------------------------------------------------------------------------------------------------------------------

while True:

    # ----------------------------------------------------------------------------------------------------------------------

    """
    # cv2.VideoCapture.read() ---> 读取视频，在while中循环读取视频的frame
    # vs.read() ---> 得到两个参数，其中ret是布尔值，如果读取帧是正确的则返回True，
    # 如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
    # 第一个参数为False的时候，if not grabbed --- True --- 循环结束，

    """
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    # ----------------------------------------------------------------------------------------------------------------------

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    """
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    # 将图片构建成一个blob，设置图片尺寸，然后执行一次，YOLO前馈网络计算，最终获取边界框和相应概率
    # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False) -------
    # 这个函数是用来读取图片的接口，其中参数很重要，会直接影响到模型的检测效果，前面几个参数与模型训练的时候对图片
    # 进行预处理有关系。其中最后一个参数是blob = cv2.dnn.blobFromImage()，swapRB，是选择是否交换R与B颜色通道，
    # 一般用opencv读取caffe的模型就需要将这个参数设置为false， 读取tensorflow的模型， 则默认选择True即可，读取
    # dnn(暗网)时，即yolo网络，swapRB设置为True，需要交换R与B颜色通道。
    # layerOutputs = net.forward(ln) ---> YOLO前馈网络计算，最终获取边界框和相应概率
    # 
    """

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    """
    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    # boxes ---> 用于存放识别物体的框的一些信息 ---> 框的左上角横坐标x和纵坐标y以及框的高h和宽w
    # confidences ---> 表示识别是某种物体的可信度
    # classIDs ---> 表示识别物体归属于哪一类 ---> ['person', 'bicycle', 'car', 'motorbike'....]

    """

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            """
            # extract the class ID and confidence (i.e., probability) of the current object detection
            # scores = detection[5:] ---> [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  							   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #    						   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
            #  							   0. 0. 0. 0. 0. 0. 0. 0.]
            #							  ....
            # classID = np.argmax(scores) ---> 得到物体的类别ID
            # confidence = scores[classID] ---> 得到是某种物体的置信度

            """

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            """
            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            # 只保留置信度大于某值的边界框 ---> 这里设置的是50%的置信度，也就是可能性大于百分之五十就可以显示出来
            # 如果图片的质量较差或者视频的质量较差， 那么可以将置信度的边界值调整低一些，这样也许可以识别更多的物体

            """

            if confidence > 0.5:
                """
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                # 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是边界框的中心坐标以及边界框的宽度和高度
                # np.array ---> 生成一个数组 
                # box.astype("int") ---> 使用 astype("int") 对上述 array 进行强制类型转换
                # centerX ---> 框的中心点横坐标， centerY ---> 框的中心点纵坐标
                # width ---> 框的宽度， height ---> 框的高度

                """

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                """
                # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                # x = int(centerX - (width / 2))  ---> 计算边界框的左上角的横坐标
                # y = int(centerY - (height / 2)) ---> 计算边界框的左上角的纵坐标

                """

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                """
                # update our list of bounding box coordinates, confidences,and class IDs
                # boxes.append([x, y, int(width), int(height)]) ---> 将边框的信息添加到列表boxes
                # confidences.append(float(confidence)) ---> 将识别出是某种物体的置信度添加到列表confidences
                # classIDs.append(classID) ---> 将识别物体归属于哪一类的信息添加到列表classIDs

                """

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    """
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes ensure at least one detection exists
    # 使用非极大值抑制方法抑制弱、重叠边界框 ---> 对于检测出结果的优化处理
    # 应用非最大值抑制可以抑制明显重叠的边界框，只保留最自信（confidence）的边界框，NMS还确保我们没有任何冗余或无关的边界框。
    # 利用OpenCV内置的NMS DNN模块实现即可实现非最大值抑制 ，所需要的参数是边界 框、 置信度、以及置信度阈值和NMS阈值。
    # cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"]) --- 第一个参数是存放边界框的列表
    # 第二个参数是存放置信度的列表，第三个参数是自己设置的置信度，第四个参数是关于threshold（阈值）

    """

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    """
    # if len(idxs) > 0 ---> 假设存在至少一个检测结果，就循环用非最大值抑制确定idx ---> for i in idxs.flatten()。
    # 然后，我们使用随机类颜色在图像上绘制边界框和文本。最后，显示结果图像，直到用户按下键盘上的任意键。--- cv2.waitKey(0)
    #
    """
    if len(idxs) > 0:
        for i in idxs.flatten():
            """
            # extract the bounding box coordinates
            # 得到边框左上角的横坐标纵坐标，边框的高度和宽度 

            """
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ################################################################################

    """
    # cv2.VideoWriter_fourcc --- 视频的编码格式
    # cv2.VideoWriter --- 视频输出 --- 将视频文件写入到硬盘
    # 第一个参数是视频输出的地址，第二个参数是视频的编码格式，第三个参数是视频的帧率，
    # 第四个参数是视频宽和高

    """

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output/out.mp4", fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    writer.write(frame)
    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------------------------------------------------------

"""
# 释放资源，退出程序

"""
writer.release()
vs.release()

# ----------------------------------------------------------------------------------------------------------------------
