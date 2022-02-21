# cv2.__version__ 4.5.2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import datetime

# 颜色范围定义
low_blue_bgr = np.array([100, 50, 0])
high_blue_bgr = np.array([255, 150, 67])
low_blue_hsv = np.array([100, 43, 46])
high_blue_hsv = np.array([124, 255, 255])
low_green_bgr = np.array([120, 150, 60])
high_green_bgr = np.array([210, 210, 210])
low_green_hsv = np.array([35, 43, 46])
high_green_hsv = np.array([77, 255, 255])

# 模板匹配类
class wordsTemplate:
    def __init__(self, path):
        # 准备模板
        def getWordsList(names, low, high):
            words_list = []
            for i in range(low, high):
                # 将模板文件路径存放在列表中
                word = self.readDirectory(path + '/'+ names[i])
                words_list.append(word)
            return words_list
        self.names = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁',
            '青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']
        # 获得中文模板列表（只匹配车牌的第一个字符）（港澳台车牌不同）
        self.chinese_words_list = getWordsList(self.names, 34, 64)
        # 获得英文模板列表（只匹配车牌的第二个字符）
        self.letter_list = getWordsList(self.names, 10, 34)
        # 获得英文和数字模板列表（匹配车牌后面的字符）
        self.letter_number_list = getWordsList(self.names, 0, 34)
    
    # 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
    def readDirectory(self, directory_name):
        refer_img_list = []
        for filename in os.listdir(directory_name):
            refer_img_list.append(directory_name + "/" + filename)
        return refer_img_list

    # 读取一个模板地址与图片进行匹配，返回得分
    def templateScore(self, template, image):
        # 将模板进行格式转换
        template_img = cv2.imdecode(np.fromfile(template, dtype = np.uint8), 1)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
        # 模板图像阈值化处理——获得黑白图
        _, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
        image_ = image.copy()
        # 获得待检测图片的尺寸
        height, width = image_.shape
        # 将模板resize至与图像一样大小
        template_img = cv2.resize(template_img, (width, height))
        # 模板匹配，返回匹配得分
        result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF_NORMED)
        return result[0][0]

    # 单个字符匹配，index选择匹配范围
    def wordMatch(self, word_image, index):
        best_score = []
        words_list = []
        # index分为三类：0：汉字，1：字母，2：字母和数字
        index = min(index, 2)
        # 对应在names中的开始位置
        start = [34, 10, 0][index]
        # 对应模板列表
        words_list = [self.chinese_words_list, self.letter_list, self.letter_number_list][index]
        for words in words_list:
            score = []
            for word in words:
                result = self.templateScore(word, word_image)
                score.append(result)
            # 每个字符对应的最佳得分
            best_score.append(max(score))
        # 具有最大最佳得分的字符
        i = best_score.index(max(best_score))
        r = self.names[start + i]
        return r

# license plate recognition
class LPR:
    def __init__(self, img_path, template):
        self.color = ""
        self.min_lic_area = 50000        # 车牌的最小面积
        self.img = cv2.imread(img_path)
        w, h = self.img.shape[:2]
        # self.img = cv2.resize(self.img, (4000, int(4000/h*w)), interpolation=cv2.INTER_AREA) 
        self.template = template
    
    def getLicense(self):
        # 识别车牌颜色并获得掩膜，蓝色好识别，绿色易受树叶等干扰
        def getColorMask(img, min_lic_area):
            color = 'blue'
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img, low_blue_bgr, high_blue_bgr)
            mask_hsv = cv2.inRange(img_hsv, low_blue_hsv, high_blue_hsv)
            # mask = mask * mask_hsv
            if (np.sum(mask/255) < min_lic_area):
                mask = cv2.inRange(img, low_green_bgr, high_green_bgr)
                color = 'green'
            return color, mask
        
        # 对掩膜进行形态学处理，使其闭合
        def getBinaryImage(mask):
            kernel_small = np.ones((3, 3))
            kernel_big = np.ones((30, 30))
            img_bin = cv2.GaussianBlur(mask, (5, 5), 0) # 高斯平滑
            img_bin = cv2.erode(img_bin, kernel_small, iterations=5) # 腐蚀5次
            img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_big, iterations=10) # 闭操作
            img_bin = cv2.GaussianBlur(img_bin, (5, 5), 0) # 高斯平滑
            _, img_bin = cv2.threshold(img_bin, 100, 255, cv2.THRESH_OTSU) # 二值化
            return img_bin
        
        # 根据轮廓的最小外接矩形找到车牌
        def findCorrectBox(img_bin, min_lic_area):
            contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            ret = None, None
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                area = cv2.contourArea(contour)
                w, h = rect[1]
                if w < h:
                    w, h = h, w
                if area < min_lic_area:
                    continue
                scale = w / h
                # 长宽比满足条件，且面积与其最小外接矩形相差不大
                if scale > 2.5 and scale < 5.5 and area/w/h > 0.7:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    img_cp = self.img.copy()
                    cv2.drawContours(img_cp, [box], 0, (0, 0, 255), 10)
                    # cv2.imshow("", img_cp)
                    ret = rect, box
            return ret

        # 透视变换使车牌变正
        def adjustAngle(img, rect, box):
            # 获取四个顶点坐标
            left_point_x = np.min(box[:, 0])
            right_point_x = np.max(box[:, 0])
            top_point_y = np.min(box[:, 1])
            bottom_point_y = np.max(box[:, 1])
            left_point_y = sorted(box[:, 1][np.where(box[:, 0] == left_point_x)])[0]
            right_point_y = sorted(box[:, 1][np.where(box[:, 0] == right_point_x)])[-1]
            top_point_x = sorted(box[:, 0][np.where(box[:, 1] == top_point_y)])[-1]
            bottom_point_x = sorted(box[:, 0][np.where(box[:, 1] == bottom_point_y)])[0]
            # 上下左右四个点坐标
            vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [right_point_x, right_point_y]])
            w, h = rect[1]
            # 默认拍摄到的车牌的竖边近似竖直
            # 畸变情况1
            if w < h:
                new_right_point_x = vertices[0, 0]
                new_right_point_y = int(vertices[1, 1] - (vertices[0, 0]- vertices[1, 0]) / (vertices[3, 0] - vertices[1, 0]) * (vertices[1, 1] - vertices[3, 1]))
                new_left_point_x = vertices[1, 0]
                new_left_point_y = int(vertices[0, 1] + (vertices[0, 0] - vertices[1, 0]) / (vertices[0, 0] - vertices[2, 0]) * (vertices[2, 1] - vertices[0, 1]))
                # 校正后的四个顶点坐标
                point_set_1 = np.float32([[440, 0],[0, 0],[0, 140],[440, 140]])
            # 畸变情况2
            elif w > h:
                new_right_point_x = vertices[1, 0]
                new_right_point_y = int(vertices[0, 1] + (vertices[1, 0] - vertices[0, 0]) / (vertices[3, 0] - vertices[0, 0]) * (vertices[3, 1] - vertices[0, 1]))
                new_left_point_x = vertices[0, 0]
                new_left_point_y = int(vertices[1, 1] - (vertices[1, 0] - vertices[0, 0]) / (vertices[1, 0] - vertices[2, 0]) * (vertices[1, 1] - vertices[2, 1]))
                # 校正后的四个顶点坐标
                point_set_1 = np.float32([[0, 0],[0, 140],[440, 140],[440, 0]])
            # 校正前平行四边形四个顶点坐标
            new_box = np.array([(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]), (new_right_point_x, new_right_point_y)])
            point_set_0 = np.float32(new_box)
            # 变换矩阵
            mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
            # 投影变换
            lic = cv2.warpPerspective(img, mat, (440, 140))
            return lic

        # 车牌二值化
        def getBinaryLicense(lic, color):
            lic_gray = cv2.cvtColor(lic, cv2.COLOR_BGR2GRAY)
            _, lic_bin = cv2.threshold(lic_gray, 0, 255, cv2.THRESH_OTSU)
            if color == 'green':
                lic_bin = cv2.bitwise_not(lic_bin)
            return lic_bin

        self.color, mask = getColorMask(self.img, self.min_lic_area)
        # print(self.color)
        img_binary = getBinaryImage(mask)
        rect, box = findCorrectBox(img_binary, self.min_lic_area)
        lic = adjustAngle(self.img, rect, box)
        # cv2.imshow('', lic)
        self.lic_binary = getBinaryLicense(lic, self.color)
    
    def getWords(self):
        # 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
        def findWaves(threshold, histogram):
            up_point = -1  # 上升点
            is_peak = False
            if histogram[0] > threshold:
                up_point = 0
                is_peak = True
            wake_peak_list = [] # (上升点，下降点)的List，表示一个波峰的开始和结束
            for i, x in enumerate(histogram):
                if is_peak and x < threshold:
                    if i - up_point > 2:
                        is_peak = False
                        wake_peak_list.append((up_point, i))
                elif not is_peak and x >= threshold:
                    is_peak = True
                    up_point = i
            if is_peak and up_point != -1 and i - up_point > 4:
                wake_peak_list.append((up_point, i))
            return wake_peak_list

        # 根据找出的波峰，分隔图片，从而得到逐个字符图片
        def separateCard(img, waves):
            part_cards = []
            for wave in waves:
                part_cards.append(img[:, wave[0]:wave[1]]) # 只分割列，从wave[0]列到wave[1]列
            return part_cards
        
        # 对疑似车牌区域进行逐行累加统计，目的是为了去掉上下边框
        def removeHorizontal(img_bin):
            x_histogram = np.sum(img_bin, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2
            wave_peak_list = findWaves(x_threshold, x_histogram)
            wave = max(wave_peak_list, key=lambda x: x[1] - x[0])
            img_bin = img_bin[wave[0]:wave[1]]
            return img_bin

        def getPosList(img_bin):
            # 对疑似车牌区域进行逐列累加，其目的是为了分割各个字符（汉字，分隔符，字母，数字）
            y_histogram = np.sum(img_bin, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 7  # U和0要求阈值偏小，否则U和0会被分成两半
            wave_peak_list = findWaves(y_threshold, y_histogram)
            wave = max(wave_peak_list, key=lambda x: x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            
            # 组合分离汉字
            cur_dis = 0
            for i, wave in enumerate(wave_peak_list):
                if wave[1]-wave[0]+cur_dis > max_wave_dis*0.6:   #需要满足一定的宽度
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            # 前几个(汉字的部分)组合到一起
            if i > 0:
                wave = (wave_peak_list[0][0], wave_peak_list[i][1])
                wave_peak_list = wave_peak_list[i + 1:]
                wave_peak_list.insert(0, wave)

            # 去除可能的其他干扰
            waves_copy = wave_peak_list.copy()
            wave_peak_list = []
            for i, wave in enumerate(waves_copy):
                if wave[1] - wave[0] > max_wave_dis / 2:
                    wave_peak_list.append(wave)
            return wave_peak_list
        
        self.lic_binary = removeHorizontal(self.lic_binary)
        wave_list = getPosList(self.lic_binary)
        self.words = separateCard(self.lic_binary, wave_list)

        plt.figure()
        for i, img in enumerate(self.words):
            plt.subplot(1, len(self.words), i+1)
            plt.imshow(img, cmap='gray')
        # plt.show()
    
    def getTemplateResult(self):
        results = []
        # 依次识别每个字符
        for index, image in enumerate(self.words):
            results.append(self.template.wordMatch(image, index))
        self.result = "".join(results)

    def showResult(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        w, h = img_rgb.shape[:2] 
        img_rgb = cv2.resize(img_rgb, (800, int(w * 800 / h)))
        img_pil = Image.fromarray(img_rgb)
        # PIL图片上打印汉字
        draw = ImageDraw.Draw(img_pil)  # 图片上打印
        font = ImageFont.truetype("simhei.ttf", 100, encoding="utf-8")
        draw.text((0, 0), self.result,  (255, 0, 0), font=font)
        # PIL图片转cv2 图片
        cv2charimg = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("result", cv2charimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        self.getLicense()
        self.getWords()
        self.getTemplateResult()
        print("Matching Result: ", self.result)
        # self.showResult()

template_model = wordsTemplate("./refer1")
# L = LPR("./images/easy/1-2.jpg", template_model)
# L.run()


# 读取测试数据
for level in ("easy", "medium", "difficult"):
    directory_name = "./images/" + level
    for filename in os.listdir(directory_name):
        print(filename)
        start = datetime.datetime.now()
        img_path = directory_name + "/" + filename
        L = LPR(img_path, template_model)
        L.run()
        end = datetime.datetime.now()
        print((end - start).seconds)


