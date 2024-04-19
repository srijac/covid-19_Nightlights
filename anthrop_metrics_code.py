import pandas as pd
import glob
import numpy as np
import os
from itertools import combinations
import math

def cv_pv_cdi(num_val):
    m=np.mean(num_val)
    std=np.std(num_val)
    cv=-100
    pv=-100
    cdi=-100
    if m!=0:
        cv=std/m
    else:
        print(f'cv not calculated default-100, as m =0')
    n = len(num_val)
    c = n * (n - 1) / 2.
    # get all pairs in l
    pairs = list(combinations(num_val, 2))
    df_pairs = pd.DataFrame(pairs).rename(columns={0: 'zi', 1: 'zj'})
    # get max / min for each pair
    max_val = df_pairs.max(axis=1)
    min_val = df_pairs.min(axis=1)
    df_pairs['max_z'] = max_val
    df_pairs['min_z'] = min_val

    # absolute difference between z
    df_pairs['diff_z_abs'] = (df_pairs['zi'] - df_pairs['zj']).abs()

    df_pairs['ratio_diff_max'] = df_pairs['diff_z_abs'] / df_pairs['max_z']
    df_pairs['one_minus_ratio'] = 1. - df_pairs['min_z'] / df_pairs['max_z']
    pv = (1/c)*(df_pairs['ratio_diff_max'].sum())
    df_ts = pd.DataFrame(num_val)
    
    
    df_ts['con_ratio']=(df_ts['obs_ntl']/df_ts['obs_ntl'].shift())
    df_ts['con_ratio_log']=(np.log(df_ts['con_ratio']))
    cdi=((np.abs(df_ts['con_ratio_log'])).sum())/(n-1)
    
    return cv, pv, cdi


def daily_change_metrics(df):
 column_names = ['dates','pred_ntl','obs_ntl','decision','confidence']
 df_time_step = pd.DataFrame(columns=column_names)
 df_time_step['dates']=df.iloc[:,0]
 df_time_step['pred_ntl']=df.iloc[:,1]
 df_time_step['obs_ntl']=df.iloc[:,2]
 df_time_step['magnitude']=np.abs(df_time_step.iloc[:,1]-df_time_step.iloc[:,2])
 df_time_step['percent_change']=((df_time_step.iloc[:,1]-df_time_step.iloc[:,2])/(df_time_step.iloc[:,1]))*100
 df_time_step['decision']=df.iloc[:,3]
 df_time_step['direction']=np.where((df.iloc[:,3]>0), (np.sign(df.iloc[:,2]-df.iloc[:,1])),0)
 df_time_step['confidence']=df.iloc[:,4]
 
 return df_time_step
 
def get_ch_seg(avg_bin, start):
    while (avg_bin[start]>0):
        #print('searching, elem', start,avg_bin[start])
        seg_end=start
        start+=1
        if (start>=avg_bin.shape[0]):
            break
    return seg_end
    
def get_ch_seg_seq(bin_avg):
    rec=np.zeros((bin_avg.shape[0],1))
    ch_seg=[]
    
    i=0
    while i in np.arange(0,bin_avg.shape[0]):
        if (bin_avg[i]>0) & (i<bin_avg.shape[0]-1):# change point before upto last but 1
            flag=1#hitting else after flag is 1 indicates after ch seg
            end=get_ch_seg(bin_avg,i)
            ch_seg.append(tuple((i,end)))
            i=end+1#starting from next 0
        elif (bin_avg[i]>0) & (i==bin_avg.shape[0]-1):#ch starts at the last time step; so append(i,i)
            end=i
            ch_seg.append(tuple((i,end)))
            i=end+1
        else:# no ch, keep looking for next ch pt
            i+=1
    return ch_seg
 
def ch_summary_methods(bin_ens_avg, avg_pred_ens,start_idx,norm_ts_mm):
    #----------------change summary for each method separately (if change point, degree/ magnitude, direction (obs-pred))------
    ch_summary=np.zeros((1,bin_ens_avg.shape[0],3))# models(ann,cnn,ensx lstm) x time-steps x (ch pt, degree, direction)
    for i in np.arange(0,(bin_ens_avg.shape[0])):
        if bin_ens_avg[i]==1:
            ch_summary[0,i,0]=1
            ch_summary[0,i,1]=np.abs(avg_pred_ens[i]-norm_ts_mm[i])# change to/ add % decline
            ch_summary[0,i,2]=-(avg_pred_ens[i]-norm_ts_mm[i])# -ve for reduction, +ve for increase
            
    return ch_summary
 
def get_inflection(ch_seg, avg_pred,ch_summary,method ):
    #inflection, change rates for each method/ ensemble?
    inflPt_rate=np.zeros((len(ch_seg),3))# inflection point, change start rate, change decay rate
    for idx,seg_idx in enumerate(ch_seg):
        #start end of each segment
        seg_idx_0=ch_seg[idx][0]
        seg_idx_1=ch_seg[idx][1]
        seg=ch_summary[method,seg_idx_0:+seg_idx_1+1,1]#3 for lstm replace with weighted ensemble,(2) shows change direction
        max_idx=np.argmax(np.abs(seg))
        #print('max:', max_idx, max_idx+seg_idx_0)
        inflPt_rate[idx,0]=max_idx+seg_idx_0#relative  position within segment
        #inflPt_rate[idx,3]=max_idx+seg_idx_0#abs  position 
        if seg_idx_0>0:
            rate_start=(avg_pred[seg_idx_0+max_idx]-avg_pred[seg_idx_0-1])/(max_idx+1)
            inflPt_rate[idx,1]=rate_start
        elif seg_idx_0==0:
            if max_idx>0:#ALSMOST SAME AS LAST CASE; NOT NEEDED
                rate_start=(avg_pred[seg_idx_0+max_idx]-avg_pred[seg_idx_0])/(max_idx)
                inflPt_rate[idx,1]=rate_start
            else:
                rate_start=math.nan
                inflPt_rate[idx,1]=rate_start
        if (seg_idx_1)<(ch_summary.shape[1]-1):
            rate_end=(avg_pred[seg_idx_1+1]-avg_pred[seg_idx_0+max_idx])/(seg_idx_1+1-(seg_idx_0+max_idx))
            inflPt_rate[idx,2]=rate_end
        elif (seg_idx_1)==(ch_summary.shape[1]-1):
            if (seg_idx_1-(seg_idx_0+max_idx))>0:
                rate_end=(avg_pred[seg_idx_1]-avg_pred[seg_idx_0+max_idx])/(seg_idx_1-(seg_idx_0+max_idx))
                inflPt_rate[idx,2]=rate_end
            else:
                rate_end=math.nan
                inflPt_rate[idx,2]=rate_end
    
    return inflPt_rate
    
def get_seg_metrics(ch_seg,inflPt_rate,inversed_obs,inversed_pred, d_full, column_names_dates, start_idx):
    df_seg = pd.DataFrame(columns=column_names_dates)
    seg_idx=np.zeros((len(ch_seg),4))#start, end, inflection; start rate; end rate; overall diff (pred-obs); %change in seg
    seg_start_list=[]
    seg_end_list=[]
    seg_inf_list=[]
    for idx,seg_idx_val in enumerate(ch_seg):
        
        seg_idx_0=ch_seg[idx][0]
        seg_idx_1=ch_seg[idx][1]
        
        seg_idx[idx,0]=inflPt_rate[idx,1]
        seg_idx[idx,1]=inflPt_rate[idx,2]
        seg_idx[idx,2]=np.sum(np.abs(inversed_obs[seg_idx_0-1]-inversed_obs[seg_idx_0:seg_idx_1+1]))
        seg_idx[idx,3]=(np.sum(np.abs(inversed_obs[seg_idx_0-1]-inversed_obs[seg_idx_0:seg_idx_1+1])))/(seg_idx_1-seg_idx_0+1)
        
        
        seg_start_list.append(d_full[seg_idx_0])
        seg_end_list.append(d_full[seg_idx_1])
        seg_inf_list.append(d_full[int(inflPt_rate[idx,0])])
        
    df_seg['start']=seg_start_list
    df_seg['end']=seg_end_list
    df_seg['inflection']=seg_inf_list
    df_seg['start_rate']=seg_idx[:,0]
    df_seg['end_rate']=seg_idx[:,1]
    df_seg['total_severity']=seg_idx[:,2]#np.sum(np.abs(inversed_obs[seg_idx_0-1,0]-inversed_obs[seg_idx_0:seg_idx_1,0]))
    df_seg['average_severity']=seg_idx[:,3]
    
    return  df_seg

#DISRUPTION METRICS
for k in glob.glob('anthropause_edits/single_d/fua*.csv'):
 f1=rf'{k}'
 f_name=f1.split('\\')[1]
 df=pd.read_csv(f'anthropause_edits\\single_d\\{f_name}')
 fua=f_name.split('_')[1]
 tile=f_name.split('_')[2]
 #daily change metrics
 df_time_step=daily_change_metrics(df)
 if not os.path.exists(f"anthropause_edits\disruption"):
    os.mkdir(f"anthropause_edits\disruption")
 if not os.path.exists(f"anthropause_edits\disruption\daily_change\\"):
    os.mkdir(f"anthropause_edits\disruption\daily_change\\")
 df_time_step.to_csv(f"anthropause_edits\\disruption\\daily_change\\fua_{fua}_{tile}_time_step_change_metrics.csv",index=False,float_format='%10.6f')#w_dir_dc
 
 #uncertainty metrics
 yr=(pd.to_datetime(df['dates'])).dt.year
 pos_2019=np.asarray(np.where(yr==2019))
 start_idx=0
 end_idx=pos_2019[0,-1]
 cv, pv, cdi=cv_pv_cdi(df.iloc[0:end_idx,2])
 if not os.path.exists(f"anthropause_edits\disruption\city_uncertainty\\"):
    os.mkdir(f"anthropause_edits\disruption\city_uncertainty\\")
 with open(f"anthropause_edits\\disruption\\city_uncertainty\\fua_{fua}_{tile}_U_Output.txt", "w") as text_file:
  text_file.write("coeff of var: %0.6f \n" % (cv))
  text_file.write("pv: %0.6f \n" % (pv))
  text_file.write("cdi: %0.6f " % (cdi))
  text_file.close()
 
 #change segments
 ch_summary=ch_summary_methods(df.iloc[:,3],df.iloc[:,1],start_idx, df.iloc[:,2])
    
 ch_seg_ens=get_ch_seg_seq(df.iloc[:,3])
 
 infl_rate_ens=get_inflection(ch_seg_ens, df.iloc[:,2],ch_summary,0)
 
 column_names_dates=['start','end','inflection','start_rate','end_rate','total_severity','average_severity']
 df_ens=get_seg_metrics(ch_seg_ens,infl_rate_ens,df.iloc[:,2],df.iloc[:,1],df.iloc[:,0],column_names_dates, start_idx)
 if not os.path.exists(f"anthropause_edits\disruption\change_segment\\"):
    os.mkdir(f"anthropause_edits\disruption\change_segment\\")
 df_ens.to_csv(f"anthropause_edits\\disruption\\change_segment\\fua_{fua}_{tile}_segment_change_metrics.csv",index=False,float_format='%10.6f')#w_dir_dc

#RECOVERY METRICS
for k in glob.glob('anthropause_edits/stl_data/fua*.csv'):
 f1=rf'{k}'
 f_name=f1.split('\\')[1]
 rec=pd.read_csv(f'anthropause_edits\\stl_data\\{f_name}')
 fua=f_name.split('_')[1]
 tile=f_name.split('_')[2]
 column_names_rec=['dates','ntl_rollingAverage','forecast','difference', 'state']
 df_rec = pd.DataFrame(columns=column_names_rec)
 df_rec['dates']= rec.iloc[:,0]
 df_rec['ntl_rollingAverage']=rec.iloc[:,1]
 df_rec['forecast']=rec.iloc[:,2]
 df_rec['difference']=rec.iloc[:,2]-rec.iloc[:,1]
 df_rec['state']= df_rec['difference'].apply(lambda x: 1 if x<0 else 0) 
 #recovery
 if not os.path.exists(f"anthropause_edits\\recovery"):
    os.mkdir(f"anthropause_edits\\recovery")
 df_rec.to_csv(f"anthropause_edits\\recovery\\fua_{fua}_{tile}_recovery_metrics.csv",index=False,float_format='%10.6f')#w_dir_dc
 
    
 