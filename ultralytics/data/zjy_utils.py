import numpy as np
import cv2


def receptiveField(img, R=3, r=1, fac_r=-1, fac_R=6):
    """
    感受野滤波函数：生成环形卷积核，对图像进行卷积滤波处理
    :param img: 输入图像
    :param R: 外圆半径，卷积核的最大作用半径
    :param r: 内圆半径，区分内外区域的半径
    :param fac_r: 内圆区域的权重系数
    :param fac_R: 外圆区域的权重系数
    :return: 滤波后的输出图像
    """
    # img1 = np.float32(img)  # 注释代码：将图像转换为32位浮点型

    # 生成二维网格坐标，尺寸为(2R+1)x(2R+1)，对应卷积核大小
    x, y = np.meshgrid(np.arange(1, R * 2 + 2), np.arange(1, R * 2 + 2))
    # 计算网格中每个点到卷积核中心的欧氏距离
    dis = np.sqrt((x - (R + 1)) ** 2 + (y - (R + 1)) ** 2)
    # 标记距离小于等于内圆半径的区域
    flag1 = (dis <= r)
    # 标记距离大于内圆半径、小于等于外圆半径的区域
    flag2 = np.logical_and(dis > r, dis <= R)
    # 根据标记区域和对应权重生成卷积核
    kernal = flag1 * fac_r + flag2 * fac_R
    # 对卷积核进行归一化处理，保证权重和为1
    kernal = kernal / kernal.sum()
    # 使用自定义卷积核对图像进行二维卷积滤波
    out = cv2.filter2D(img, -1, kernal)
    return out


def SimOTM(img):
    """
    图像三通道融合函数：原始图+均值模糊图+感受野滤波图 合并为3通道图像
    :param img: 输入单通道图像
    :return: 3通道融合后的图像
    """
    # 对图像进行3x3窗口的均值模糊处理
    blur = cv2.blur(img, (3, 3))
    # 对图像进行感受野滤波处理
    rec = receptiveField(img)
    # 合并三个单通道图像为三通道图像
    result = cv2.merge([img, blur, rec])
    return result

def SimOTMBBS(img):
    """
    图像三通道融合函数：原始图+双份均值模糊图 合并为3通道图像
    :param img: 输入单通道图像
    :return: 3通道融合后的图像
    """
    # 对图像进行3x3窗口的均值模糊处理
    blur = cv2.blur(img, (3, 3))
    # 合并原始图+模糊图+模糊图为三通道图像
    result = cv2.merge([img, blur, blur])
    return result

def SimOTMSSS(img):
    """
    图像三通道融合函数：三份原始图像 合并为3通道图像（适配TIF 16位图像）
    :param img: 输入单通道图像
    :return: 3通道融合后的图像
    """
    # 合并三份原始单通道图像为三通道图像
    result = cv2.merge([img, img, img])
    return result

def enhance_brightness_or_contrast(image, target_gray_value, brightness_alpha=1.5, contrast_alpha=1.0, beta=0):
    """
    自适应亮度/对比度增强函数：根据图像平均灰度自动调整亮度或对比度
    :param image: 输入图像
    :param target_gray_value: 目标平均灰度值
    :param brightness_alpha: 亮度增强系数（未使用）
    :param contrast_alpha: 对比度调整系数
    :param beta: 亮度偏移值
    :return: 增强后的图像
    """
    # 计算输入图像的平均灰度值
    gray_value = np.mean(image)
    # 如果图像平均灰度大于等于目标值，仅调整对比度
    if gray_value >= target_gray_value:
        enhanced_image = cv2.convertScaleAbs(image, alpha=contrast_alpha, beta=beta)
    # 如果图像平均灰度小于目标值，补偿亮度差值
    else:
        avg_diff = target_gray_value - gray_value
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.0, beta=avg_diff)
    return enhanced_image

def SimOTMBrights(img):
    """
    图像三通道融合函数：与SimOTM功能一致，原始图+均值模糊图+感受野滤波图 合并为3通道图像
    :param img: 输入单通道图像
    :return: 3通道融合后的图像
    """
    # 对图像进行3x3窗口的均值模糊处理
    blur = cv2.blur(img, (3, 3))
    # 对图像进行感受野滤波处理
    rec = receptiveField(img)
    # 合并三个单通道图像为三通道图像
    result = cv2.merge([img, blur, rec])
    return result