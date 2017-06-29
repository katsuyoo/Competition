# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import operator
from scipy import stats as sts

# 拼图验证码
def read_train():
    fo=open('/home/frank/data/mouse/dsjtzs_txfz_training.txt')
    train_data=fo.readlines()
    data=[] #轨迹数据
    p_data=[] #目标点数据
    label=[] #标签
    for line in train_data:
        line=line.split(' ')
        pos=line[-2].split(',')
        pos[0]=float(pos[0])
        pos[1]=float(pos[1])
        p_data.append(tuple(pos))
        label.append(int(line[-1][0]))
        #print line
        node_list=line[1].split(';')[:-1]
        #print node_list
        sample=[]
        for node in node_list:
            info=node.split(',')
            for i in range(3):
                info[i]=float(info[i])
            sample.append(tuple(info))
        sample.sort(key=operator.itemgetter(2))
        data.append(tuple(sample))
    data=np.array(data)
    return data,p_data,label

def read_test():
    fo=open('/home/frank/data/mouse/dsjtzs_txfz_test1.txt')
    test_data=fo.readlines()
    id=[]
    data=[]
    p_data=[]
    for line in test_data:
        line=line.split(' ')
        id.append(line[0])
        pos=line[-2].split(',')
        pos[0]=float(pos[0])
        pos[1]=float(pos[1])
        p_data.append(tuple(pos))
        #print line
        node_list=line[1].split(';')[:-1]
        #print node_list
        sample=[]
        for node in node_list:
            info=node.split(',')
            for i in range(3):
                info[i]=float(info[i])
            sample.append(tuple(info))
        sample.sort(key=operator.itemgetter(2))
        data.append(tuple(sample))
    data=np.array(data)
    return id,data,p_data

def dis(x1,y1,x2,y2):
    return (x1-x2)**2 + (y1-y2)**2

def gen_train(data,p_data,label):
    print 'train'
    train_df=extract_features(data, p_data,label,1)
    train_df.to_csv('/home/frank/data/mouse/train.csv',index=None)

def gen_test(id,data,p_data):
    print 'test'
    test_df=extract_features(data, p_data,id,0)
    test_df.to_csv('/home/frank/data/mouse/test.csv', index=None)


def extract_features(data,p_data,ex,flag):

    new_features=['tot_time','x_start_speed','x_twenty_per_speed','x_median_speed','x_eighty_per_speed',
                  'x_end_speed','x_avg_speed','x_sum_distance','tan','start_x','stop_x']

    tot_time=[]

    avg_speed=[]
    distance=[]
    sum_distances=[]
    tans=[]
    start_x=[]
    start_y=[]
    stop_x=[]
    stop_y=[]

    xv_end=[]


    x_distance = []  #终点位置和目标位置x轴的距离
    xv_var=[] #速度的方差
    ax_var=[] #加速度的方差
    x_var=[] #水平位移的方差
    y_var=[] #垂直位移的方差
    d_var=[] #倾斜位移的方差
    time_var=[] #时间段的方差
    tan_var=[] #正玄值方差

    xv_skew=[] #速度的偏度
    ax_skew=[] #加速度的偏度
    x_skew=[] #水平位移的偏度
    y_skew=[] #垂直位移的偏度
    d_skew=[] #倾斜位移
    time_skew=[] #时间段的偏度

    xv_kurt=[] #速度的峰度
    ax_kurt=[] #加速度的峰度
    x_kurt=[] #水平位移的峰度
    y_kurt=[] #垂直位移的峰度
    d_kurt=[] #倾斜位移的峰度
    time_kurt=[] #时间段的峰度

    # 标准差
    xv_std=[]
    ax_std=[]
    x_std=[]
    y_std=[]
    d_std=[]
    time_std=[]

    ax_cummin=[] #加速度的累计最小值
    ax_cummax=[] #加速度的累计最大值

    xv_max=[] #速度最大值
    ax_max=[] #加速度最大值
    x_max=[]
    y_max=[]
    d_max=[]
    time_max=[] #时间段最大值

    ax_avg=[] #加速度平均值
    xv_avg=[] #速度平均值
    x_avg=[]
    y_avg=[]
    d_avg=[]
    time_avg=[]

    cnt=[] #轨迹点数目


    # reverse_ax=[] #加速度的倒数
    # reverse_ax_var=[] #加速度倒数的方差

    n=len(data)
    for i in range(n):
        xv = []  # 速度
        ax = []  # 加速度
        x_shift = []  # 水平位移
        y_shift = []  # 垂直位移
        d=[] # 倾斜位移
        time = []  # 时间段
        ta=[]  # x/y

        m=len(data[i])

        if m<5:
            ex.pop(i)
            continue
        cnt.append(m)
        tot_time.append(data[i][-1][-1]-data[i][0][-1])

        if data[i][m-1][-1]==data[i][m-2][-1]:
            xv_end.append(0)
        else:
            xv_end.append(abs(data[i][m-1][0]-data[i][m-2][0])/(data[i][m-1][-1]-data[i][m-2][-1]))


        start_x.append(data[i][0][0])
        stop_x.append(data[i][-1][0])
        start_y.append(data[i][0][1])
        stop_y.append(data[i][-1][1])

        sum_distance=sum_time=tan=max_tan_difference=0.0
        x_sum_distance=0.0
        for j in range(m-1):
            sum_distance+=dis(data[i][j][0],data[i][j][1],data[i][j+1][0],data[i][j+1][1])
            x_sum_distance+=abs(data[i][j+1][0]-data[i][j][0])
            sum_time+=data[i][j+1][-1]-data[i][j][-1]
            tan+=abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])
            ta.append(abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0]))
            if abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])>max_tan_difference:
                max_tan_difference=abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])

        avg_speed.append(sum_distance/sum_time)
        sum_distances.append(sum_distance)
        distance.append(dis(data[i][-1][0],data[i][-1][1],p_data[i][0],p_data[i][1]))
        tans.append(tan)

        # x_avg_speed.append(x_sum_distance / sum_time)
        # x_sum_distances.append(x_sum_distance)

        # new feature engineering
        for j in range(m - 1):
            if data[i][j + 1][-1] == data[i][j][-1]:
                continue
            else:
                time.append(data[i][j + 1][-1] - data[i][j][-1])
                xv.append(abs(data[i][j + 1][0] - data[i][j][0]) / (data[i][j + 1][-1] - data[i][j][-1]))
                x_shift.append(abs(data[i][j + 1][0] - data[i][j][0]))
                y_shift.append(abs(data[i][j + 1][1] - data[i][j][1]))
                d.append(abs(dis(data[i][j][0],data[i][j][1],data[i][j+1][0],data[i][j+1][1])))


        k=len(xv)
        for j in range(k - 1):
            ax.append((xv[j + 1] - xv[j]) / (time[j+1]+time[j])*2)

        xv_var.append(np.var(xv))
        ax_var.append(np.var(ax))
        x_var.append(np.var(x_shift))
        y_var.append(np.var(y_shift))
        time_var.append(np.var(time))
        tan_var.append(np.var(ta))
        d_var.append(np.var(d))

        xv_std.append(np.std(xv))
        ax_std.append(np.std(ax))
        x_std.append(np.std(x_shift))
        y_std.append(np.std(y_shift))
        time_std.append(np.std(time))
        d_std.append(np.std(d))

        xv_skew.append(sts.skew(np.array(xv)))
        ax_skew.append(sts.skew(np.array(ax)))
        x_skew.append(sts.skew(np.array(x_shift)))
        y_skew.append(sts.skew(np.array(y_shift)))
        time_skew.append(sts.skew(np.array(time)))
        d_skew.append(sts.skew(np.array(d)))

        xv_kurt.append(sts.kurtosis(xv))
        ax_kurt.append(sts.kurtosis(ax))
        x_kurt.append(sts.kurtosis(x_shift))
        y_kurt.append(sts.kurtosis(y_shift))
        time_kurt.append(sts.kurtosis(time))
        d_kurt.append(sts.kurtosis(d))

        xv_max.append(np.max(xv))
        x_max.append(np.max(x_shift))
        y_max.append(np.max(y_shift))
        ax_max.append(np.max(ax))
        time_max.append(np.max(time))
        d_max.append(np.max(d))

        xv_avg.append(np.mean(xv))
        ax_avg.append(np.mean(ax))
        x_avg.append(np.mean(x_shift))
        y_avg.append(np.mean(y_shift))
        time_avg.append(np.mean(time))
        d_avg.append(np.mean(d))

        x_distance.append(abs(data[i][-1][0] - p_data[i][0]))

    df=pd.DataFrame({'tan':tans,'cnt':cnt,
                     'xv_var':xv_var,'ax_var':ax_var,'x_var':x_var,'y_var':y_var,'time_var':time_var,'tan_var':tan_var,'d_var':d_var,
                     'xv_std':xv_std,'ax_std':ax_std,'x_std':x_std,'y_std':y_std,'time_std':time_std,'d_std':d_std,
                     'xv_skew':xv_skew,'ax_skew':ax_skew,'x_skew':x_skew,'y_skew':y_skew,'time_skew':time_skew,'d_skew':d_skew,
                     'xv_kurt':xv_kurt,'ax_kurt':ax_kurt,'x_kurt':x_kurt,'y_kurt':y_kurt,'time_kurt':time_kurt,'d_kurt':d_kurt,
                     'xv_avg': xv_avg, 'ax_avg': ax_avg,'x_avg':x_avg,'y_avg':y_avg,'time_avg':time_avg,'d_avg':d_avg,
                     'xv_max':xv_max,'x_max':x_max,'y_max':y_max,'ax_max':ax_max,'time_max':time_max,'d_max':d_max})
    if flag==1:
        df['label'] = ex
    else:
        df['id'] = ex
    return df


train_data,train_p_data,train_label=read_train()
test_id,test_data,test_p_data=read_test()

gen_train(train_data,train_p_data,train_label)
gen_test(test_id,test_data,test_p_data)