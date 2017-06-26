# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import operator


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

def extract_train_features(data,p_data,label):
    # tot_time:总时间, start_speed:开始速度, twenty_per_speed:20%位置的速度, median_speed:中间位置的速度, avg_speed:平均速度
    # distance:终点与目标点的距离, sum_distance:总位移, tan:相邻两点的正玄值之差, start_x: 开始位置X坐标
    features=['tot_time','start_speed','twenty_per_speed','median_speed','eighty_per_speed','end_speed','avg_speed',
              'distance','sum_distance','tan','start_x','start_y','stop_x','stop_y']
    new_features=['tot_time','x_start_speed','x_twenty_per_speed','x_median_speed','x_eighty_per_speed',
                  'x_end_speed','x_avg_speed','x_sum_distance','tan','start_x','stop_x']

    tot_time=[]
    start_speed=[]
    twenty_per_speed=[]
    median_speed=[]
    eighty_per_speed=[]
    end_speed=[]
    avg_speed=[]
    distance=[]
    sum_distances=[]
    tans=[]
    start_x=[]
    start_y=[]
    stop_x=[]
    stop_y=[]
    max_tan_differences=[]

    x_start_speed=[]
    x_twenty_per_speed=[]
    x_median_speed=[]
    x_eighty_per_speed=[]
    x_end_speed=[]
    x_avg_speed=[]
    x_sum_distances=[]



    xv_var=[] #速度的方差
    ax_var=[] #加速度的方差
    x_var=[] #水平位移的方差
    time_var=[] #时间段的方差

    # reverse_ax=[] #加速度的倒数
    # reverse_ax_var=[] #加速度倒数的方差

    n=len(data)
    for i in range(n):
        xv = []  # 速度
        ax = []  # 加速度
        x_shift = []  # 水平位移
        time = []  # 时间段
        m=len(data[i])
        if m<5:
            label.pop(i)
            continue
        tot_time.append(data[i][-1][-1]-data[i][0][-1])
        if data[i][1][-1]==data[i][0][-1]:
            start_speed.append(0)
        else:
            start_speed.append(dis(data[i][1][0],data[i][1][1],data[i][0][0],data[i][0][1])/(data[i][1][-1]-data[i][0][-1]))
        if data[i][m/2+1][-1]==data[i][m/2][-1]:
            median_speed.append(0)
        else:
            median_speed.append(dis(data[i][m/2][0],data[i][m/2][1],data[i][m/2+1][0],data[i][m/2+1][1])/(data[i][m/2+1][-1]-data[i][m/2][-1]))
        if data[i][m*4/5-1][-1]==data[i][m*4/5][-1]:
            eighty_per_speed.append(0)
        else:
            eighty_per_speed.append(dis(data[i][m*4/5][0],data[i][m*4/5][1],data[i][m*4/5-1][0],data[i][m*4/5-1][1])/(data[i][m*4/5][-1]-data[i][m*4/5-1][-1]))
        if data[i][m/5][-1]==data[i][m/5+1][-1]:
            twenty_per_speed.append(0)
        else:
            twenty_per_speed.append(dis(data[i][m/5+1][0],data[i][m/5+1][1],data[i][m/5][0],data[i][m/5][1])/(data[i][m/5+1][-1]-data[i][m/5][-1]))
        if data[i][m-1][-1]==data[i][m-2][-1]:
            end_speed.append(0)
        else:
            end_speed.append(dis(data[i][m-1][0],data[i][m-1][1],data[i][m-2][0],data[i][m-2][1])/(data[i][m-1][-1]-data[i][m-2][-1]))
        #
        #
        # if data[i][1][-1] == data[i][0][-1]:
        #     x_start_speed.append(0)
        # else:
        #     x_start_speed.append(abs(data[i][1][0]-data[i][0][0])/(data[i][1][-1]-data[i][0][-1]))
        # if data[i][m / 5][-1] == data[i][m / 5 + 1][-1]:
        #     x_twenty_per_speed.append(0)
        # else:
        #     x_twenty_per_speed.append(abs(data[i][m / 5 + 1][0]-data[i][m / 5][0])/(data[i][m / 5 + 1][-1]-data[i][m / 5][-1]))
        # if data[i][m/2+1][-1]==data[i][m/2][-1]:
        #     x_median_speed.append(0)
        # else:
        #     x_median_speed.append(abs(data[i][m/2+1][0]-data[i][m/2][0])/(data[i][m/2+1][-1]-data[i][m/2][-1]))
        # if data[i][m*4/5-1][-1]==data[i][m*4/5][-1]:
        #     x_eighty_per_speed.append(0)
        # else:
        #     x_eighty_per_speed.append(abs(data[i][m*4/5][0]-data[i][m*4/5-1][0])/(data[i][m*4/5][-1]-data[i][m*4/5-1][-1]))
        if data[i][m-1][-1]==data[i][m-2][-1]:
            x_end_speed.append(0)
        else:
            x_end_speed.append(abs(data[i][m-1][0]-data[i][m-2][0])/(data[i][m-1][-1]-data[i][m-2][-1]))


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
            if abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])>max_tan_difference:
                max_tan_difference=abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])

        avg_speed.append(sum_distance/sum_time)
        sum_distances.append(sum_distance)
        distance.append(dis(data[i][-1][0],data[i][-1][1],p_data[i][0],p_data[i][1]))
        tans.append(tan)
        max_tan_differences.append(max_tan_difference)

        # x_avg_speed.append(x_sum_distance / sum_time)
        # x_sum_distances.append(x_sum_distance)

        # new feature engineering
        for j in range(m - 1):
            time.append(data[i][j+1][-1]-data[i][j][-1])
            if data[i][j + 1][-1] == data[i][j][-1]:
                continue
            else:
                sd = abs(data[i][j + 1][0] - data[i][j][0]) / (data[i][j + 1][-1] - data[i][j][-1])
                xv.append([sd, data[i][j + 1][-1]])
        for j in range(m):
            x_shift.append(data[i][j][0])
        k=len(xv)
        for j in range(k - 1):
            ax.append((xv[j + 1][0] - xv[j][0]) / (xv[j+1][1] - xv[j][1]))
            # reverse_ax.append((xv[j+1][1] - xv[j][1]) / (xv[j + 1][0] - xv[j][0]))

        xv_var.append(np.var(xv))
        ax_var.append(np.var(ax))
        x_var.append(np.var(x_shift))
        time_var.append(np.var(time))

    # print len(tot_time),len(start_speed),len(avg_speed),len(distance),len(sum_distances),len(tans),len(start_x),len(max_tan_differences),len(label)

    # train_df=pd.DataFrame({'tot_time':tot_time,'x_start_speed':x_start_speed,'x_twenty_per_speed':x_twenty_per_speed,
    #                        'x_median_speed':x_median_speed,'x_eighty_per_speed':x_eighty_per_speed,'x_end_speed':x_end_speed,
    #                        'x_avg_speed':x_avg_speed,'x_sum_distance':x_sum_distances,'tan':tans,'start_x':start_x,
    #                        'stop_x':stop_x,'label':label})
    # train_df=pd.DataFrame({'start_speed':start_speed,'median_speed':median_speed,
    #     'eighty_per_speed':eighty_per_speed,'end_speed':end_speed,
    #     'tan':tans,'max_tan_difference':max_tan_differences,'tot_time':tot_time,'sum_distance':sum_distances,'label':label})
    # train_df=pd.DataFrame({'end_speed':end_speed,'tan':tans,'start_x':start_x,'label':label})
    train_df=pd.DataFrame({'tot_time':tot_time,'start_speed':start_speed,'median_speed':median_speed,'end_speed':end_speed,
        'avg_speed':avg_speed,'distance':distance,'sum_distance':sum_distances,'tan':tans,'start_x':start_x,
                           'xv_var':xv_var,'ax_var':ax_var,'x_var':x_var,'time_var':time_var,'label':label})
    # train_df = pd.DataFrame({'tan': tans,'xv_var':xv_var,'ax_var':ax_var,'x_var':x_var,'time_var':time_var,
    #                          'start_x':start_x,'end_speed':end_speed,'label':label})
    train_df.to_csv('/home/frank/data/mouse/train.csv',index=None)


def extract_test_features(id,data,p_data):
    features=['tot_time','start_speed','twenty_per_speed','median_speed','eighty_per_speed','end_speed','avg_speed',
              'distance','sum_distance','tan','start_x','start_y','stop_x','stop_y']

    tot_time=[]
    start_speed=[]
    twenty_per_speed=[]
    median_speed=[]
    eighty_per_speed=[]
    end_speed=[]
    avg_speed=[]
    distance=[]
    sum_distances=[]
    tans=[]
    start_x=[]
    start_y=[]
    stop_x=[]
    stop_y=[]
    max_tan_differences=[]

    x_start_speed=[]
    x_twenty_per_speed=[]
    x_median_speed=[]
    x_eighty_per_speed=[]
    x_end_speed=[]
    x_avg_speed=[]
    x_sum_distances=[]



    xv_var=[] #速度的方差
    ax_var=[] #加速度的方差
    x_var=[] #水平位移的方差
    time_var=[] #时间段的方差
    # reverse_ax=[] #加速度的倒数
    # reverse_ax_var=[] #加速度倒数的方差


    n=len(data)
    for i in range(n):
        xv = []  # 速度
        ax = []  # 加速度
        x_shift = []  # 水平位移
        time = []  # 时间段
        m=len(data[i])
        if m<5:
            id.pop(i)
            continue
        tot_time.append(data[i][-1][-1]-data[i][0][-1])
        if data[i][1][-1]==data[i][0][-1]:
            start_speed.append(0)
        else:
            start_speed.append(dis(data[i][1][0],data[i][1][1],data[i][0][0],data[i][0][1])/(data[i][1][-1]-data[i][0][-1]))
        if data[i][m/2+1][-1]==data[i][m/2][-1]:
            median_speed.append(0)
        else:
            median_speed.append(dis(data[i][m/2][0],data[i][m/2][1],data[i][m/2+1][0],data[i][m/2+1][1])/(data[i][m/2+1][-1]-data[i][m/2][-1]))
        if data[i][m*4/5-1][-1]==data[i][m*4/5][-1]:
            eighty_per_speed.append(0)
        else:
            eighty_per_speed.append(dis(data[i][m*4/5][0],data[i][m*4/5][1],data[i][m*4/5-1][0],data[i][m*4/5-1][1])/(data[i][m*4/5][-1]-data[i][m*4/5-1][-1]))
        if data[i][m/5][-1]==data[i][m/5+1][-1]:
            twenty_per_speed.append(0)
        else:
            twenty_per_speed.append(dis(data[i][m/5+1][0],data[i][m/5+1][1],data[i][m/5][0],data[i][m/5][1])/(data[i][m/5+1][-1]-data[i][m/5][-1]))
        if data[i][m-1][-1]==data[i][m-2][-1]:
            end_speed.append(0)
        else:
            end_speed.append(dis(data[i][m-1][0],data[i][m-1][1],data[i][m-2][0],data[i][m-2][1])/(data[i][m-1][-1]-data[i][m-2][-1]))
        #
        #
        # if data[i][1][-1] == data[i][0][-1]:
        #     x_start_speed.append(0)
        # else:
        #     x_start_speed.append(abs(data[i][1][0]-data[i][0][0])/(data[i][1][-1]-data[i][0][-1]))
        # if data[i][m / 5][-1] == data[i][m / 5 + 1][-1]:
        #     x_twenty_per_speed.append(0)
        # else:
        #     x_twenty_per_speed.append(abs(data[i][m / 5 + 1][0]-data[i][m / 5][0])/(data[i][m / 5 + 1][-1]-data[i][m / 5][-1]))
        # if data[i][m/2+1][-1]==data[i][m/2][-1]:
        #     x_median_speed.append(0)
        # else:
        #     x_median_speed.append(abs(data[i][m/2+1][0]-data[i][m/2][0])/(data[i][m/2+1][-1]-data[i][m/2][-1]))
        # if data[i][m*4/5-1][-1]==data[i][m*4/5][-1]:
        #     x_eighty_per_speed.append(0)
        # else:
        #     x_eighty_per_speed.append(abs(data[i][m*4/5][0]-data[i][m*4/5-1][0])/(data[i][m*4/5][-1]-data[i][m*4/5-1][-1]))
        if data[i][m-1][-1]==data[i][m-2][-1]:
            x_end_speed.append(0)
        else:
            x_end_speed.append(abs(data[i][m-1][0]-data[i][m-2][0])/(data[i][m-1][-1]-data[i][m-2][-1]))

        start_x.append(data[i][0][0])
        stop_x.append(data[i][-1][0])
        start_y.append(data[i][0][1])
        stop_y.append(data[i][-1][1])

        sum_distance=sum_time=tan=max_tan_difference=0.0
        x_sum_distance = 0.0
        for j in range(m-1):
            sum_distance += dis(data[i][j][0], data[i][j][1], data[i][j + 1][0], data[i][j + 1][1])
            x_sum_distance += abs(data[i][j + 1][0] - data[i][j][0])
            sum_time+=data[i][j+1][-1]-data[i][j][-1]
            tan+=abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])
            if abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])>max_tan_difference:
                max_tan_difference=abs(data[i][j][1]/data[i][j][0]-data[i][j+1][1]/data[i][j+1][0])

        avg_speed.append(sum_distance/sum_time)
        sum_distances.append(sum_distance)
        distance.append(dis(data[i][-1][0], data[i][-1][1], p_data[i][0], p_data[i][1]))
        tans.append(tan)
        max_tan_differences.append(max_tan_difference)

        x_avg_speed.append(x_sum_distance / sum_time)
        x_sum_distances.append(x_sum_distance)


        #new feature engineering
        for j in range(m - 1):
            time.append(data[i][j+1][-1]-data[i][j][-1])
            if data[i][j + 1][-1] == data[i][j][-1]:
                continue
            else:
                sd = abs(data[i][j + 1][0] - data[i][j][0]) / (data[i][j + 1][-1] - data[i][j][-1])
                xv.append([sd, data[i][j + 1][-1]])
        for j in range(m):
            x_shift.append(data[i][j][0])
        k=len(xv)
        for j in range(k - 1):
            ax.append((xv[j + 1][0] - xv[j][0]) / (xv[j+1][1] - xv[j][1]))
            # reverse_ax.append((xv[j+1][1] - xv[j][1]) / (xv[j + 1][0] - xv[j][0]))

        xv_var.append(np.var(xv))
        ax_var.append(np.var(ax))
        x_var.append(np.var(x_shift))
        time_var.append(np.var(time))

    # test_df=pd.DataFrame({'id':id,'tot_time':tot_time,'x_start_speed':x_start_speed,'x_twenty_per_speed':x_twenty_per_speed,
    #                    'x_median_speed':x_median_speed,'x_eighty_per_speed':x_eighty_per_speed,'x_end_speed':x_end_speed,
    #                    'x_avg_speed':x_avg_speed,'x_sum_distance':x_sum_distances,'tan':tans,'start_x':start_x,
    #                    'stop_x':stop_x})
    # test_df=pd.DataFrame({'id':id,'start_speed':start_speed,'median_speed':median_speed,
    #     'eighty_per_speed':eighty_per_speed,'end_speed':end_speed,
    #     'tan':tans,'max_tan_difference':max_tan_differences,'tot_time':tot_time,'sum_distance':sum_distances})
    # test_df=pd.DataFrame({'id':id,'end_speed':end_speed,'tan':tans,'start_x':start_x})
    test_df=pd.DataFrame({'id':id,'tot_time':tot_time,'start_speed':start_speed,'median_speed':median_speed,
                          'end_speed':end_speed,'avg_speed':avg_speed,'distance':distance,'sum_distance':sum_distances,
                          'tan':tans,'xv_var':xv_var,'ax_var':ax_var,'x_var':x_var,'time_var':time_var,
                          'start_x':start_x})
    # test_df=pd.DataFrame({'id':id,'tan':tans,'xv_var':xv_var,'ax_var':ax_var,'x_var':x_var,'time_var':time_var,
    #                       'start_x': start_x, 'end_speed': end_speed})
    test_df.to_csv('/home/frank/data/mouse/test.csv',index=None)


train_data,train_p_data,train_label=read_train()
test_id,test_data,test_p_data=read_test()

extract_train_features(train_data,train_p_data,train_label)
extract_test_features(test_id,test_data,test_p_data)