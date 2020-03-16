# -*- coding:utf-8 -*-
# author: DuanshengLiu
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
from tensorflow import keras
import numpy as np


class Window:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 350, 200))
        self.win.title("车牌识别软件---by DuanshengLiu")
        self.path = None
        # self.tag == None

        self.label1 = Label(self.win, text='原图:', font=('微软雅黑', 10))
        self.label1.place(x=0, y=0)
        self.label2 = Label(self.win, text='车牌区域:', font=('微软雅黑', 10))
        self.label2.place(x=650, y=0)
        self.label3 = Label(self.win, text='识别结果:', font=('微软雅黑', 10))
        self.label3.place(x=650, y=80)

        self.can1 = Canvas(self.win, width=605, height=355, bg='white', relief='solid', borderwidth=1)
        self.can1.place(x=35, y=5)
        self.can2 = Canvas(self.win, width=260, height=60, bg='white', relief='solid', borderwidth=1)
        self.can2.place(x=710, y=5)
        self.can3 = Canvas(self.win, width=260, height=60, bg='white', relief='solid', borderwidth=1)
        self.can3.place(x=710, y=85)

        self.button1 = Button(self.win, text='选择文件', width=10, height=1, command=self.load_show_img)
        self.button1.place(x=680, y=wh - 30)
        self.button2 = Button(self.win, text='识别车牌', width=10, height=1, command=self.display)
        self.button2.place(x=780, y=wh - 30)
        self.button3 = Button(self.win, text='清空所有', width=10, height=1, command=self.clear)
        self.button3.place(x=880, y=wh - 30)
        self.model_c = keras.models.load_model("model_c.h5")
        self.model_d = keras.models.load_model("model_d.h5")
        self.model_c_label = np.array(['京', '闽', '粤', '苏', '沪', '浙'])  # 训练集目前只有这6个省份等以后扩充
        self.model_d_label = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                                       'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                       'W', 'X', 'Y', 'Z'])

    def load_show_img(self):
        sv = StringVar()
        sv.set(askopenfilename())
        self.path = Entry(self.win, state='readonly', text=sv).get()  # 获取到了所打开的图片
        img_open = Image.open(self.path).resize((600, 350))
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can1.create_image(6, 6, image=self.img_Tk, anchor='nw')

    def display(self):
        self.can3.delete("all")  # 防止不清空所有就进行下次预测,预测的字会叠加在画板3上的问题
        if self.path == None:  # 还没选择图片就进行预测
            self.can3.create_text(47, 15, text='请选择图片', anchor='nw', font=('黑体', 27))
        else:
            self.locate_license_plate()
            if self.tag == True:
                self.can2.create_image(6, 6, image=self.license_plate_Tk, anchor='nw')
                self.predict()  # 车牌分割完进行预测
                self.can3.create_text(47, 15, text=self.pred, anchor='nw', font=('黑体', 27))
            else:
                self.can3.create_text(47, 15, text='未能识别', anchor='nw', font=('黑体', 27))

    def locate_license_plate(self):
        img = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8), -1)  # 防止从中文路径选择图片报错
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low_hsv = np.array([100, 43, 46])
        high_hsv = np.array([124, 255, 255])
        mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)  # 选取图中蓝色的区域

        kernel_small = np.ones((3, 3))
        kernel_big = np.ones((7, 7))
        img_Gas = cv2.GaussianBlur(mask, (5, 5), 50)
        img_di = cv2.dilate(img_Gas, kernel_small, iterations=5)  # 腐蚀5次
        img_close = cv2.morphologyEx(img_di, cv2.MORPH_CLOSE, kernel_big)  # 闭操作
        img_close = cv2.GaussianBlur(img_close, (5, 5), 50)  # 高斯平滑
        self.img_close_mean = img_close.mean()  # 用于判断是不是整个图片就是一张车牌

        edge = cv2.Canny(img_close, 100, 200)
        close1 = cv2.resize(img_close, (1000, 600))

        # cv2.imshow('edge',close1)
        # cv2.waitKey(0)

        # print(img_close.mean())
        # print(img_close.shape)
        contours1, hierarchy1 = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(img,contours1+contours2,-1,(0,0,255),2)
        # cv2.imshow('0',img)
        # cv2.waitKey(0)

        self.coordinates = []
        for each in contours1 + contours2:
            x, y, w, h = cv2.boundingRect(each)  # 最小外接矩形,返回的是矩形左上角坐标以及宽度和高度
            x0, y0 = each.min(axis=0)[0, 0], each.min(axis=0)[0, 1]  # 找出边缘坐标的最小值
            x1, y1 = each.max(axis=0)[0, 0], each.max(axis=0)[0, 1]  # 找出边缘坐标的最大值
            if (x1 - x0) * (y1 - y0) > 0:  # 筛选掉x1=x0或者y1=y0
                if img_close[y:y + h, x:x + w].mean() >= 180 and 2 <= w / h <= 5 and abs(
                        (x1 - x0) / (y1 - y0) - w / h) <= 1:
                    '''
                    #对于二值化图像,所选中的区域的均值在180以上符合我们想要的,
                    且车牌区域的宽高比应该在3左右,设置在(2-5)之间防止图片有变形,
                    且车牌区域的边缘坐标与最小外接矩形应该是相匹配的
                    '''

                    self.coordinates.append([y0, y1, x0, x1])
        # print(self.coordinates)

        if self.coordinates != []:
            for coor in self.coordinates:
                y0, y1, x0, x1 = coor[0], coor[1], coor[2], coor[3]
                self.license_plate = cv2.resize(img[y0:y1, x0:x1], (250, 60))  # 将车牌resize成width=250,height=60
                self.split_license_plate()  # 进行车牌字符分割
                if len(self.License) < 7:  # 分割的车牌字符少于7判定该区域不是车牌区域
                    # print('小于7,长度为%s'%len(self.License))
                    self.tag = False
                    continue
                else:  # 否则判定该区域是车牌区域
                    # print('等于7,长度为%s'%len(self.License))
                    # print("宽高比",(x1-x0)/(y1-y0))
                    self.tag = True
                    self.license_plate_Tk = ImageTk.PhotoImage(Image.fromarray(self.license_plate[:, :, ::-1]))
                    # self.license_plate_Tk用于车牌区域的显示,[:,:,::-1]是因为cv2一个像素是[B,G,R],和Imgae的[R,G,B]相反
                    break

        else:
            # 满足下面条件说明可能整个图片就是一张车牌
            if self.img_close_mean >= 180 and 2 <= img_close.shape[1] / img_close.shape[0] <= 5:
                self.tag = True
                self.license_plate = cv2.resize(img, (250, 60))  # 将车牌resize成width=250,height=60
                self.license_plate_Tk = ImageTk.PhotoImage(Image.fromarray(self.license_plate[:, :, ::-1]))
                self.split_license_plate()

            else:
                self.tag = False

    def split_license_plate(self):
        '''
            分割车牌字符函数除了真正意义的分割车牌字符，
            还有一个功能是分割的字符若小于7说明不是车牌区域，用于过滤self.coordinates中无效的坐标
        '''

        if self.img_close_mean <= 248:  # 判断图片不完全是车牌,还存在左右边框区域,所以截取[10:50,10:245],防止边框的干扰
            # print(self.img_close_mean)
            self.license_plate_ = self.license_plate[10:50, 10:245]
        else:
            # print(self.img_close_mean)
            self.license_plate_ = self.license_plate[5:55, :]
        license_plate_gray = cv2.cvtColor(self.license_plate_, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        # 阈值根据彩色车牌的均值来设置，因为每张图片的色彩有差异
        threshold_value = self.license_plate_.mean() + 20
        # 图片二值化
        license_plate_binary = cv2.threshold(license_plate_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow('0',license_plate_binary)
        # cv2.waitKey(0)

        white = []  # white记录每一列的白色像素个数
        black = []  # black记录每一列的黑色像素个数
        height = license_plate_binary.shape[0]
        width = license_plate_binary.shape[1]

        # 计算每一列的黑白色像素总和
        for i in range(width):
            w = 0  # 每一列白色像素
            b = 0  # 每一列黑色像素
            for j in range(height):
                if license_plate_binary[j][i] == 255:
                    w += 1
                if license_plate_binary[j][i] == 0:
                    b += 1
            white.append(w)
            black.append(b)

        # 寻找字符的末尾
        def find_end(start):
            end = start + 1
            for m in range(start + 1, width - 1):
                if black[m] > 0.95 * max(black):  # 如果某列黑色像素出现个数>所有列黑色像素出现次数的最大值的0.95倍，则判定为字符的结尾
                    end = m
                    break
            return end

        l = []
        n = 0
        while n <= width - 2:
            n += 1
            if white[n] > 0.05 * max(white):  # 如果某列白色像素出现个数>所有列白色像素出现个数的最大值的0.05倍，则判断为字符的开头
                start = n
                end = find_end(start)
                n = end
                # 排除车牌中的的·(均值一般<30)
                if license_plate_binary[0:height, start:end].mean() >= 30:
                    # 将start和end扩展,否则字符占据了整个图片,与训练集不太一致
                    if 5 <= end - start <= 10:  # 该字符可能是1,所以比较窄
                        # print(license_plate_binary[0:height, start:end].mean())
                        start_ = start - 12 if start >= 12 else start
                        l.append([start_, end + 12])  # 保存车牌字符的首尾

                    if end - start > 10:
                        # print(license_plate_binary[0:height, start:end].mean())
                        start_ = start - 3 if start >= 3 else start  # 防止start-3<0报错
                        l.append([start_, end + 3])  # 保存车牌字符的首尾
        # print('l',l)
        self.License = []
        if len(l) == 9:  # 首尾都是带边框的字符,舍弃
            l = l[1:8]
        if len(l) == 8:  # 首尾有一端出现带边框的字符
            start_1, end_1 = l[0][0], l[0][1]  # 第1个车牌字符的左右坐标
            start_2, end_2 = l[1][0], l[1][1]  # 第2个车牌字符的左右坐标
            start__2, end__2 = l[-2][0], l[-2][1]  # 第7个车牌字符的左右坐标
            start__1, end__1 = l[-1][0], l[-1][1]  # 第8个车牌字符的左右坐标
            if start_2 < end_1:  # 如果第2个字符的开头小于第1个字符的结尾,删除第1个字符
                l.pop(0)
            if start__1 < end__2:  # 如果第8个字符的开头小于倒数第7个字符的结尾,删除第8个字符
                l.pop(-1)
            if len(l) == 8:  # 若len(l)还是为8
                l = l[1:8]
        if len(l) == 6:
            # 可能漏了最后一位,原因是原始车牌图片是已经裁剪过的,最后一位没有边框后未能获得
            l.append([l[-1][1], l[-1][1] + 33])

        j = 0
        if len(l) == 7:  # 首尾无干扰,正好为7位
            for s_e in l:
                j += 1
                start, end = s_e[0], s_e[1]
                lic_binary = license_plate_binary[0:height, start:end]
                lic_binary = cv2.resize(lic_binary, (32, 40))
                self.License.append(lic_binary)
                # cv2.imshow('0', lic_binary)
                # cv2.waitKey(0)
        # print('self.License', len(self.License))

    def predict(self):
        # 预测的index是形状为(1,)的ndarray,因此加[0]将其取出
        # 先预测首个字,用model_c预测中文省份
        index = self.model_c.predict_classes(self.License[0].reshape(1, 40 * 32))[0]
        self.pred = self.model_c_label[index]
        # 再预测后续字母和数字,用model_d预测字母和数字
        for i, license in enumerate(self.License[1:]):
            index = self.model_d.predict_classes(license.reshape(1, 40 * 32))[0]
            lic = self.model_d_label[index]
            if i == 0:  # 预测完第一个字母或数字后加个' · '
                lic = self.model_d_label[index] + '·'
            self.pred += lic

    def clear(self):
        self.can1.delete("all")
        self.can2.delete("all")
        self.can3.delete("all")
        self.path = None

    def closeEvent():  # 关闭前清除session(),防止'NoneType' object is not callable
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    win = Tk()
    ww = 1000  # 窗口宽设定1000
    wh = 370  # 窗口高设定370
    Window(win, ww, wh)
    win.protocol("WM_DELETE_WINDOW", Window.closeEvent)
    win.mainloop()
