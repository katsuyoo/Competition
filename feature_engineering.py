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


def cul_cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return 1
    else:
        return dot_product / ((normA*normB)**0.5)

def gen_train(data,p_data,label):
    print 'train'
    train_df=extract_features(data, p_data,label,1)
    train_df.to_csv('/home/frank/data/mouse/train.csv',index=None)

def gen_test(id,data,p_data):
    print 'test'
    test_df=extract_features(data, p_data,id,0)
    test_df.to_csv('/home/frank/data/mouse/test.csv', index=None)


def extract_features(data,p_data,ex,flag):
    #x速度均值，标准差，初始点和终止点位置，x初始速度，总时间，x,y的初始位置，y速度最大值
    # xv_avg,xv_std,y_distance,x_distance,start_x,start_y,(stop_x-start_x)+(stop_y-start_y),xv_start,tot_time,yv_max
    tot_time=[]

    start_x=[]
    start_y=[]
    stop_x=[]
    stop_y=[]
    d_tot=[]
    x_tot=[]
    y_tot=[]
    xv_start=[]
    yv_max=[]

    xv_end=[]


    x_distance = []  #终点位置和目标位置x轴的距离
    y_distance = []
    distance = [] #  终点位置和目标位置的距离


    stop_cnt=[] #  x停住点数目
    point_cnt = []  # 轨迹点数目
    t_cnt = [] # t停住点数目
    y_cnt = [] #  y坐标变化次数
    yy_sum=[] # y坐标差值之和

    xv_var=[] #速度的方差
    ax_var=[] #加速度的方差
    x_var=[] #水平位移的方差
    y_var=[] #垂直位移的方差
    d_var=[] #倾斜位移的方差
    time_var=[] #时间段的方差
    tan_var=[] #正弦值方差(angle of ab)
    cos_var=[] #余弦值方差(angle of abc)
    dif_var=[]

    xv_skew=[] #速度的偏度
    ax_skew=[] #加速度的偏度
    x_skew=[] #水平位移的偏度
    y_skew=[] #垂直位移的偏度
    d_skew=[] #倾斜位移
    time_skew=[] #时间段的偏度
    tan_skew=[]
    cos_skew=[]
    dif_skew=[]

    xv_kurt=[] #速度的峰度
    ax_kurt=[] #加速度的峰度
    x_kurt=[] #水平位移的峰度
    y_kurt=[] #垂直位移的峰度
    d_kurt=[] #倾斜位移的峰度
    time_kurt=[] #时间段的峰度
    tan_kurt=[]
    cos_kurt=[]
    dif_kurt=[]

    # 标准差
    xv_std=[]
    ax_std=[]
    x_std=[]
    y_std=[]
    d_std=[]
    time_std=[]
    tan_std=[]
    cos_std=[]
    dif_std=[]

    ax_cummin=[] #加速度的累计最小值
    ax_cummax=[] #加速度的累计最大值

    xv_max=[] #速度最大值
    ax_max=[] #加速度最大值
    x_max=[]
    y_max=[]
    d_max=[]
    time_max=[] #时间段最大值
    tan_max=[]
    cos_max=[]
    dif_max=[]

    ax_avg=[] #加速度平均值
    xv_avg=[] #速度平均值
    x_avg=[]
    y_avg=[]
    d_avg=[]
    time_avg=[]
    tan_avg=[]
    cos_avg=[]
    dif_avg=[]

    tan_sum=[]
    cos_sum=[]
    dif_sum=[]
    d_sum=[]
    y_sum=[]



    # reverse_ax=[] #加速度的倒数
    # reverse_ax_var=[] #加速度倒数的方差

    n=len(data)
    cc=0
    for i in range(n):
        xv = []  # x速度
        yv = []  # y速度
        ax = []  # 加速度
        x_shift = []  # 水平位移
        y_shift = []  # 垂直位移
        d=[] # 倾斜位移
        time = []  # 时间段
        tan=[]  # 正弦值
        cos=[] # 余弦值
        dif=[] # 两个点坐标差值之比

        m=len(data[i])

        if m<2:
            ex.pop(i-cc)
            cc+=1
            continue
        gg=1
        for j in range(m - 1):
            if data[i][j + 1][0] != data[i][j][0]:
                gg=0
                break
        if gg==1:
            ex.pop(i)
            continue

        point_cnt.append(m)

        if data[i][1][-1] == data[i][0][-1]:
            xv_start.append(0)
        else:
            xv_start.append(abs(data[i][1][0]-data[i][0][0])/(data[i][1][-1]-data[i][0][-1]))

        if data[i][m-1][-1]==data[i][m-2][-1]:
            xv_end.append(0)
        else:
            xv_end.append(abs(data[i][m-1][0]-data[i][m-2][0])/(data[i][m-1][-1]-data[i][m-2][-1]))


        start_x.append(data[i][0][0])
        stop_x.append(data[i][-1][0])
        start_y.append(data[i][0][1])
        stop_y.append(data[i][-1][1])
        d_tot.append(dis(data[i][0][0],data[i][0][1],data[i][-1][0],data[i][-1][1]))
        x_tot.append(data[i][-1][0]-data[i][0][0])
        y_tot.append(data[i][-1][1]-data[i][0][1])
        tot_time.append(data[i][-1][-1]-data[i][0][-1])


        # new feature engineering
        cnt=0
        cnty=0
        cntt=0
        sumy=0
        for j in range(m - 1):
            sumy+=abs(data[i][j+1][1]-data[i][j][1])
            if data[i][j+1][0]!=data[i][j][0]:
                dif.append(abs((data[i][j + 1][1] - data[i][j][1])/(data[i][j + 1][0] - data[i][j][0])))
            tan.append((data[i][j + 1][0] - data[i][j][0] + 0.1) / (data[i][j + 1][1] - data[i][j][1] + 0.1))
            if data[i][j][1]!=data[i][j+1][1]:
                cnty+=1
            if data[i][j+1][0] == data[i][j][0]:
                cnt+=1
            if data[i][j + 1][-1] == data[i][j][-1]:
                cntt+=1
                continue
            else:
                time.append(data[i][j + 1][-1] - data[i][j][-1])
                xv.append(abs(data[i][j + 1][0] - data[i][j][0]) / (data[i][j + 1][-1] - data[i][j][-1]))
                yv.append(abs(data[i][j+1][1]-data[i][j][1])/(data[i][j + 1][-1] - data[i][j][-1]))
                x_shift.append(abs(data[i][j + 1][0] - data[i][j][0]))
                y_shift.append(abs(data[i][j + 1][1] - data[i][j][1]))
                d.append(abs(dis(data[i][j][0],data[i][j][1],data[i][j+1][0],data[i][j+1][1])))

        # k=len(xv)
        # for j in range(k - 1):
        #     ax.append((xv[j + 1] - xv[j]) / (time[j+1]+time[j])*2)

        # for j in range(m-2):
        #     if data[i][j+1][0]-data[i][j][0]==0 and data[i][j+1][1]-data[i][j][1]==0:
        #         continue
        #     while j<m-2 and data[i][j+2][0]-data[i][j+1][0]==0 and data[i][j+2][1]-data[i][j+1][1]==0:
        #         j+=1
        #     if j==m-2:
        #         break
        #     ab=(data[i][j+1][0]-data[i][j][0],data[i][j+1][1]-data[i][j][1])
        #     bc=(data[i][j+2][0]-data[i][j+1][0],data[i][j+2][1]-data[i][j+1][1])
        #     cos.append(cul_cos(ab,bc))
        #
        # if len(cos)==0:
        #     cos.append(0)

        stop_cnt.append(cnt)
        y_cnt.append(cnty)
        t_cnt.append(cntt)
        yy_sum.append(sumy)

        xv_var.append(np.var(xv))
        # ax_var.append(np.var(ax))
        # x_var.append(np.var(x_shift))
        # y_var.append(np.var(y_shift))
        # time_var.append(np.var(time))
        # d_var.append(np.var(d))
        # tan_var.append(np.var(tan))
        # cos_var.append(np.var(cos))
        # dif_var.append(np.var(dif))

        xv_std.append(np.std(xv))
        # ax_std.append(np.std(ax))
        # x_std.append(np.std(x_shift))
        # y_std.append(np.std(y_shift))
        # time_std.append(np.std(time))
        # d_std.append(np.std(d))
        # tan_std.append(np.std(tan))
        # cos_std.append(np.std(cos))
        # dif_std.append(np.std(dif))

        # xv_skew.append(sts.skew(np.array(xv)))
        # ax_skew.append(sts.skew(np.array(ax)))
        # x_skew.append(sts.skew(np.array(x_shift)))
        # y_skew.append(sts.skew(np.array(y_shift)))
        # time_skew.append(sts.skew(np.array(time)))
        # d_skew.append(sts.skew(np.array(d)))
        # tan_skew.append(sts.skew(np.array(tan)))
        # cos_skew.append(sts.skew(np.array(cos)))
        # dif_skew.append(sts.skew(np.array(dif)))
        #
        # xv_kurt.append(sts.kurtosis(xv))
        # ax_kurt.append(sts.kurtosis(ax))
        # x_kurt.append(sts.kurtosis(x_shift))
        # y_kurt.append(sts.kurtosis(y_shift))
        # time_kurt.append(sts.kurtosis(time))
        # d_kurt.append(sts.kurtosis(d))
        # tan_kurt.append(sts.kurtosis(tan))
        # cos_kurt.append(sts.kurtosis(cos))
        # dif_kurt.append(sts.kurtosis(dif))

        # xv_max.append(np.max(xv))
        # x_max.append(np.max(x_shift))
        # y_max.append(np.max(y_shift))
        # ax_max.append(np.max(ax))
        # time_max.append(np.max(time))
        # d_max.append(np.max(d))
        # tan_max.append(np.max(tan))
        # dif_max.append(np.max(dif))
        yv_max.append(np.max(yv))

        xv_avg.append(np.mean(xv))
        # ax_avg.append(np.mean(ax))
        # x_avg.append(np.mean(x_shift))
        # y_avg.append(np.mean(y_shift))
        # time_avg.append(np.mean(time))
        # d_avg.append(np.mean(d))
        # tan_avg.append(np.mean(tan))
        # cos_avg.append(np.mean(cos))
        # dif_avg.append(np.mean(dif))

        # tan_sum.append(np.sum(tan))
        # dif_sum.append(np.sum(dif))
        # cos_sum.append(np.sum(cos))
        # d_sum.append(np.sum(d))
        # y_sum.append(np.sum(y_shift))

        # x_distance.append(abs(data[i][-1][0] - p_data[i][0]))
        # y_distance.append(abs(data[i][-1][1]-p_data[i][1]))
        # distance.append(dis(data[i][-1][0],data[i][-1][1],p_data[i][0],p_data[i][1]))

    # df=pd.DataFrame({'point_cnt':point_cnt,'stop_cnt':stop_cnt,'xv_end':xv_end,'yy_sum':yy_sum,'tot_time':tot_time,'start_x':start_x,'t_cnt':t_cnt,'distance':distance,
    #                  'xv_var':xv_var,'xv_std':xv_std,'xv_skew':xv_skew,'xv_kurt':xv_kurt,'xv_avg':xv_avg,'xv_max':xv_max,
    #                  'ax_var': ax_var,'ax_std':ax_std,'ax_skew':ax_skew,'ax_kurt':ax_kurt,'ax_avg':ax_avg,'ax_max':ax_max,
    #                  'x_var': x_var,'x_std':x_std,'x_skew':x_skew,'x_kurt':x_kurt,'x_avg':x_avg,'x_max':x_max,
    #                  'y_var': y_var,'y_std':y_std,'y_skew':y_skew,'y_kurt':y_kurt,'y_avg':y_avg,'y_max':y_max,'y_sum':y_sum,
    #                  'time_var': time_var,'time_std':time_std,'time_skew':time_skew,'time_kurt':time_kurt,'time_avg':time_avg,'time_max':time_max,
    #                  'd_var': d_var,'d_std':d_std,'d_skew':d_skew,'d_kurt':d_kurt,'d_avg':d_avg,'d_max':d_max,'d_sum':d_sum,
    #                  'tan_var': tan_var,'tan_std':tan_std,'tan_skew':tan_skew,'tan_kurt':tan_kurt,'tan_avg':tan_avg,'tan_max':tan_max,'tan_sum':tan_sum,
    #                  'cos_var': cos_var,'cos_std':cos_std,'cos_skew':cos_skew,'cos_kurt':cos_kurt,'cos_avg':cos_avg,'cos_sum':cos_sum,
    #                  'dif_avg': dif_avg, 'dif_sum': dif_sum,'dif_var':dif_var,'dif_skew':dif_skew,'dif_kurt':dif_kurt,'dif_std':dif_std,'dif_max':dif_max
    # })


    # df=pd.DataFrame({'point_cnt':point_cnt,'stop_cnt':stop_cnt,'start_x':start_x,'xv_end':xv_end,'yy_sum':yy_sum,
    #                  'xv_std':xv_std,'xv_skew':xv_skew,'xv_kurt':xv_kurt,'xv_avg':xv_avg,'xv_max':xv_max,
    #                  'ax_std':ax_std,'ax_skew':ax_skew,'ax_kurt':ax_kurt,'ax_avg':ax_avg,'ax_max':ax_max,
    #                  'x_std':x_std,'x_skew':x_skew,'x_kurt':x_kurt,'x_avg':x_avg,
    #                  'y_std':y_std,'y_skew':y_skew,'y_kurt':y_kurt,'y_avg':y_avg,'y_max':y_max,
    #                  'time_std':time_std,'time_skew':time_skew,'time_kurt':time_kurt,'time_avg':time_avg,'time_max':time_max,
    #                  'd_std':d_std,'d_skew':d_skew,'d_kurt':d_kurt,'d_avg':d_avg,'d_max':d_max,
    #                  'tan_std':tan_std,'tan_skew':tan_skew,'tan_kurt':tan_kurt,'tan_avg':tan_avg,'tan_max':tan_max,
    #                  'cos_std':cos_std,'cos_skew':cos_skew,'cos_kurt':cos_kurt,'cos_avg':cos_avg,
    #                  'dif_avg': dif_avg, 'dif_sum': dif_sum
    # })
    # print len(xv_avg),len(xv_std),len(x_distance),len(y_distance),len(start_x),len(start_y),len(tot_time),len(d_tot),len(xv_start),len(yv_max)
    df=pd.DataFrame({'xv_avg':xv_avg,'xv_std':xv_std,'x_tot':x_tot,'y_tot':y_tot,'start_x':start_x,
                     'start_y':start_y,'tot_time':tot_time,'d_tot':d_tot,'xv_start':xv_start,'yv_max':yv_max
                     })

    if flag==1:
        df['label'] = ex
    else:
        df['id'] = ex
    return df


train_data,train_p_data,train_label=read_train()
test_id,test_data,test_p_data=read_test()

gen_train(train_data,train_p_data,train_label)
gen_test(test_id,test_data,test_p_data)