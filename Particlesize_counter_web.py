
import sys
import cv2
import tempfile
import pytesseract
import numpy as np
import pandas as pd
import streamlit as st
import scipy.ndimage as ndimage
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image





IMG_PATH = 'imgs'

st.title('粒子径計測アプリケーション')
st.caption('画像を読み込んで、粒子径を表示します。')
#画像入れ
image_path = None
mag = None

#左側配置##    
threshold = st.sidebar.slider('スケール読み取り閾値(いらない)', 0, 255, 230)
cirlim = st.sidebar.slider('円形度の閾値(%)', 0, 100, 80)
##倍率読み込み
bai = False
if st.sidebar.checkbox('スケール読み込み手動'):
    bai = True
    mag = st.sidebar.number_input('倍率', min_value=1, max_value=10000, value=50, step=1)
##閾値
iki = False
if st.sidebar.checkbox('粉末読み込み閾値手動(いらない)'):
    iki = True
    ikiti = st.sidebar.slider('白黒の閾値', 0, 255, 160)
##大きさ最小
saisyou = False
if st.sidebar.checkbox('最小値'):
    saisyou = True
    thremin = st.sidebar.number_input('切り捨て最小値', min_value=1, max_value=10000, value=1, step=1)
##大きさ最大
saidai = False
if st.sidebar.checkbox('最大値'):
    saidai = True
    thremax = st.sidebar.number_input('切り捨て最大値', min_value=1, max_value=10000, value=500, step=1)
###########


#画像入れ
image_path = st.file_uploader("画像を入れてください", type=['jpg', 'jpeg', 'png'])
if image_path != None:
    #読み込みファイル表示
    #st.image(image_path)

    #########そのままだと読み込めないので一時保存する
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_path.read())
    #読み込み画像ファイル
    #image = Image.open(temp_file.name)
    image = cv2.imread(temp_file.name)
    #st.text(type(image))
    #temp_file.close()
    #読み込みファイル表示
    #st.image(image)
    #########

    #倍率部分の画像切り取り
    im_crop = image[900: 945, 190 : 340]

    #im_crop.save('crop.jpg', quality=95)
    #st.image(im_crop)

    ###
    #倍率自動のみ
    if bai == False:
        #スケール読み取り
        ret, img_thresh = cv2.threshold(im_crop, threshold, 255, cv2.THRESH_BINARY)
        #画像から数値識別かつ数値以外の文字を除外
        mag = pytesseract.image_to_string(img_thresh, lang='eng', config='--psm 6 --oem 1 -c tessedit_char_whitelist="0123456789."').strip()
        #5を9と認識するので置換
        mag = (mag.replace('9','5'))
        st.text("スケール")
        st.text(mag)
    ###

    #メイン部分の画像切り取り
    im_crop2 = image[1: 887, 1: 1279]

    img = cv2.cvtColor(im_crop2, cv2.COLOR_BGR2GRAY)
    img8bit = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    img_disp = cv2.cvtColor(img8bit, cv2.COLOR_GRAY2BGR)

    #画像の前処理(ぼかし)
    img_blur = cv2.GaussianBlur(img,(5,5),0)

    #2値画像を取得
    if iki == True:
        ret,th = cv2.threshold(img_blur,0,255,ikiti)
    if iki == False:
        ret,th = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #モルフォロジー変換(膨張)
    kernel = np.ones((3,3),np.uint8)
    th = cv2.dilate(th,kernel,iterations = 1)

    #画像を保存
    #cv2.imwrite('thresholds.png', th)
    #Fill Holes処理
    th_fill = ndimage.binary_fill_holes(th).astype(int) * 255

    #境界検出と描画
    cnts,hierarchy = cv2.findContours(th_fill.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_cnt = cv2.drawContours(im_crop, cnts, -1, (0,255,255), 1)


    #面積、円形度、等価直径を求める。
    Areas = []
    Circularities = []
    Eq_diameters = []
    #条件満たした書き出すやつ
    dataC = []
    dataD = []

    # 輪郭の点の描画
    for cnt in cnts:
        #面積(px*px)
        area = cv2.contourArea(cnt)
        Areas.append(area)

        #円形度
        arc = cv2.arcLength(cnt, True)
        val = 4 * np.pi * area / (arc * arc)
        Circularities.append(val)   

        ###(仮)
        #mag = 200
        #等価直径(px)
        eq_diameter = np.sqrt(4*area/np.pi) * float(mag) / 104
        Eq_diameters.append(eq_diameter)

        # 輪郭の矩形領域
        x,y,w,h = cv2.boundingRect(cnt)
        # 円形度の描画
        cv2.putText(img_disp, f"C={val:.3f}", (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
        # 直径の描画
        cv2.putText(img_disp, f"D={eq_diameter:.1f}", (x, y+18), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
        # 円らしい領域（円形度が0.85以上）を囲う
        if val > float(cirlim) / 100:
            #端のやつを排除
            if x > 2 and 1297 > x + w:
                if y > 2 and 887 > y + h:
                    if saisyou == True:
                        if eq_diameter > thremin:
                            if saidai == True:
                                if eq_diameter < thremax:
                                    cv2.rectangle(img_disp,(x-5,y-5),(x+w+5,y+h+5),(255,255,255),2) # 少し外側を囲う
                                    #条件満たすやつ
                                    dataC.append(val)
                                    dataD.append(eq_diameter)
                                    tempC = round(val, 2)
                                    tempD = round(eq_diameter, 2)
                                    st.sidebar.text(f'円形度:{tempC}, 直径:{tempD}')
                            else:
                                cv2.rectangle(img_disp,(x-5,y-5),(x+w+5,y+h+5),(255,255,255),2) # 少し外側を囲う
                                #条件満たすやつ
                                dataC.append(val)
                                dataD.append(eq_diameter)
                                tempC = round(val, 2)
                                tempD = round(eq_diameter, 2)
                                st.sidebar.text(f'円形度:{tempC}, 直径:{tempD}')
                    else:
                        if saidai == True:
                            if eq_diameter < thremax:
                                cv2.rectangle(img_disp,(x-5,y-5),(x+w+5,y+h+5),(255,255,255),2) # 少し外側を囲う
                                #条件満たすやつ
                                dataC.append(val)
                                dataD.append(eq_diameter)
                                tempC = round(val, 2)
                                tempD = round(eq_diameter, 2)
                                st.sidebar.text(f'円形度:{tempC}, 直径:{tempD}')
                        else:
                            cv2.rectangle(img_disp,(x-5,y-5),(x+w+5,y+h+5),(255,255,255),2) # 少し外側を囲う
                            #条件満たすやつ
                            dataC.append(val)
                            dataD.append(eq_diameter)
                            tempC = round(val, 2)
                            tempD = round(eq_diameter, 2)
                            st.sidebar.text(f'円形度:{tempC}, 直径:{tempD}')
    st.image(img_disp)
    
    #####csv出力#####################
    df = pd.DataFrame({
    'Circularities': dataC,
    'Diameters': dataD
    })
    csv = df.to_csv().encode('SHIFT-JIS')
    st.download_button(label='Data Download', 
                   data=csv, 
                   file_name='Particlesize_data.csv',
                   mime='text/csv',
                   )







