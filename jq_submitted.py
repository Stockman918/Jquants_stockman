# -*- coding: utf-8 -*-



import io
import os
import pickle
import numpy as np
import pandas as pd
import dfply as dp
import lightgbm as lgb
import re
from sklearn.model_selection import GridSearchCV

#インプットデータセットの名称を取得
def get_inputs(dataset_dir):

    inputs = {
        "stock_list": f"{dataset_dir}/stock_list.csv.gz",
        "stock_price": f"{dataset_dir}/stock_price.csv.gz",
        "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
        # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
        "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
    }
    return inputs

#インプットデータを取得する関数
def get_dataset(inputs):
    
    dfs = {}
    for k, v in inputs.items():
        dfs[k] = pd.read_csv(v)
        # DataFrameのindexを設定します。
        if k == "stock_price":
            dfs[k].loc[:, "datetime"] = pd.to_datetime(
                dfs[k].loc[:, "EndOfDayQuote Date"]
            )
            dfs[k].set_index("datetime", inplace=True)
        elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
            dfs[k].loc[:, "datetime"] = pd.to_datetime(
                dfs[k].loc[:, "base_date"]
            )
            dfs[k].set_index("datetime", inplace=True)
    return dfs

#特徴量を取得する関数
def get_feat(dfs,st = "2017-01-01",ed = "2019-12-31",skip=False,test_flag=False,comp=False):
    
    sp = dfs['stock_price'] 
    a = pd.merge(sp,dfs['stock_list'],on=["Local Code"],how="inner") >> dp.filter_by(dp.X['prediction_target']==True)
    a['EndOfDayQuote Date'] = a['EndOfDayQuote Date'].str.replace('/','-')
    a = pd.merge(a,dfs['stock_labels'],left_on=["EndOfDayQuote Date","Local Code"],right_on=["base_date","Local Code"],how="inner")
    a['tv'] = a['EndOfDayQuote Volume']*a['EndOfDayQuote VWAP']
    a = a >> dp.select('base_date','Local Code','EndOfDayQuote High','EndOfDayQuote Low','EndOfDayQuote ExchangeOfficialClose','EndOfDayQuote Open',
              'EndOfDayQuote PercentChangeFromPreviousClose','label_high_20', 'label_low_20','label_high_10','label_low_10',
              'label_high_5', 'label_low_5','IssuedShareEquityQuote IssuedShare','17 Sector(name)','tv',
              'Section/Products','EndOfDayQuote CumulativeAdjustmentFactor')
    
    a.rename(columns={'base_date':'ymd','Local Code':'code','EndOfDayQuote High':'high','EndOfDayQuote Low':'low',
                      'EndOfDayQuote ExchangeOfficialClose':'close_official','EndOfDayQuote Open':'open',
                      'EndOfDayQuote PercentChangeFromPreviousClose':'pchange_from_pre','label_high_20':'high_20',
                      'label_low_20':'low_20','label_high_10':'high_10','label_low_10':'low_10',
                      'label_high_5':'high_5', 'label_low_5':'low_5','IssuedShareEquityQuote IssuedShare':'num_share','17 Sector(name)':'sector17_name',
                      'tv':'tv','Section/Products':'section','EndOfDayQuote CumulativeAdjustmentFactor':'cum_adjust'},inplace=True)
    
    a = a.replace(np.nan,0)
    
    a = a >> dp.group_by(dp.X.code) >> dp.mutate(s25=dp.X.close_official.rolling(25).mean(),\
                                                s75=dp.X.close_official.rolling(75).mean(),atv=dp.X.tv.rolling(20).mean()) >> \
                         dp.ungroup() >> \
    dp.mutate(sig=dp.if_else((dp.X.s25>dp.X.s75)&(dp.X.close_official>dp.X.s25),6,\
                            dp.if_else((dp.X.s25>dp.X.s75)&(dp.X.close_official>dp.X.s75),5,\
                                      dp.if_else((dp.X.s25>dp.X.s75),4,\
                                                dp.if_else((dp.X.s75>dp.X.s25)&(dp.X.close_official>dp.X.s75),3,\
                                                          dp.if_else((dp.X.s75>dp.X.s25)&(dp.X.close_official>dp.X.s25),2,1))))),\
                         s25d=dp.X.close_official/dp.X.s25-1,s75d=dp.X.close_official/dp.X.s75-1)
    
    datas = a >> dp.mutate(cap=dp.X.num_share*dp.X.close_official) >> \
    dp.group_by(dp.X.code) >> dp.mutate(ret_o=dp.lead(dp.X.open)/dp.X.close_official-1) >> \
                dp.ungroup() >> dp.group_by(dp.X.ymd) >> dp.mutate(m_ret_o=dp.mean(dp.X.ret_o),s_ret_o=dp.sd(dp.X.ret_o),\
                           m_high_20=dp.mean(dp.X.high_20),s_high_20=dp.sd(dp.X.high_20),\
                           m_low_20=dp.mean(dp.X.low_20),s_low_20=dp.sd(dp.X.low_20),\
                           m_high_10=dp.mean(dp.X.high_10),s_high_10=dp.sd(dp.X.high_10),\
                           m_low_10=dp.mean(dp.X.low_10),s_low_10=dp.sd(dp.X.low_10),\
                           m_high_5=dp.mean(dp.X.high_5),s_high_5=dp.sd(dp.X.high_5),\
                           m_low_5=dp.mean(dp.X.low_5),s_low_5=dp.sd(dp.X.low_5)) \
                           >> dp.ungroup() >> \
                           dp.mutate(reto_s=(dp.X.ret_o-dp.X.m_ret_o)/dp.X.s_ret_o,\
                                     high20_s=(dp.X.high_20-dp.X.m_high_20)/dp.X.s_high_20,\
                                     low20_s=(dp.X.low_20-dp.X.m_low_20)/dp.X.s_low_20,\
                                     high10_s=(dp.X.high_10-dp.X.m_high_10)/dp.X.s_high_10,\
                                     low10_s=(dp.X.low_10-dp.X.m_low_10)/dp.X.s_low_10,\
                                     high5_s=(dp.X.high_5-dp.X.m_high_5)/dp.X.s_high_5,\
                                     low5_s=(dp.X.low_5-dp.X.m_low_5)/dp.X.s_low_5)
    
    datas = datas.replace([np.inf,-np.inf],np.nan)
    datas = datas.fillna(0)
    
    datas = datas >> dp.group_by(dp.X.code) >> dp.mutate(RVOL=dp.X.pchange_from_pre.rolling(20).std(),\
                                 RVOL60=dp.X.pchange_from_pre.rolling(60).std()) >> \
    dp.ungroup() >> dp.group_by(dp.X.code) >> dp.mutate(RVOL_20=dp.lead(dp.X.RVOL,20)) >> dp.ungroup() >> \
    dp.select(['ymd','code','high_20','low_20','high_10','low_10','high_5','low_5','sig','s25d','s75d','cap','atv',\
                  'reto_s','RVOL','RVOL60','RVOL_20','high20_s','low20_s','high10_s','low10_s','high5_s','low5_s','section'])
    
    
    b = dfs['stock_fin']
    b = pd.merge(b,dfs['stock_list'],on=["Local Code"],how="inner") >> dp.filter_by(dp.X['prediction_target']==True)
    b = pd.merge(b,dfs['stock_price'],left_on=["base_date","Local Code"],right_on=["EndOfDayQuote Date","Local Code"],how="left")
    
    b = b >> dp.select('base_date','Local Code','Result_FinancialStatement FiscalPeriodEnd','Result_FinancialStatement ModifyDate',
                       'Result_FinancialStatement ReportType','Result_FinancialStatement NetSales','Result_FinancialStatement NetIncome',
                       'Result_FinancialStatement OrdinaryIncome','Result_FinancialStatement OperatingIncome','Forecast_FinancialStatement FiscalPeriodEnd',
                       'Forecast_FinancialStatement NetSales','Forecast_FinancialStatement OperatingIncome','Forecast_FinancialStatement OrdinaryIncome',
                       'Forecast_FinancialStatement NetIncome','Result_FinancialStatement AccountingStandard','Result_FinancialStatement TotalAssets',
                       'Result_FinancialStatement NetAssets','IssuedShareEquityQuote IssuedShare','EndOfDayQuote ExchangeOfficialClose',
                       'Forecast_Dividend AnnualDividendPerShare','Forecast_FinancialStatement ReportType')
    
    b.rename(columns={'base_date':'ymd','Local Code':'code','Result_FinancialStatement FiscalPeriodEnd':'fy','Result_FinancialStatement ModifyDate':'modify_ymd',
                       'Result_FinancialStatement ReportType':'quarter','Result_FinancialStatement NetSales':'netsales','Result_FinancialStatement NetIncome':'net_income',
                       'Result_FinancialStatement OrdinaryIncome':'orm','Result_FinancialStatement OperatingIncome':'opm',
                       'Forecast_FinancialStatement FiscalPeriodEnd':'fy_forecast',
                       'Forecast_FinancialStatement NetSales':'netsales_forecast','Forecast_FinancialStatement OperatingIncome':'opm_forecast',
                       'Forecast_FinancialStatement OrdinaryIncome':'orm_forecast','Forecast_FinancialStatement NetIncome':'net_income_forecast',
                       'Result_FinancialStatement AccountingStandard':'accounting_standard','Result_FinancialStatement TotalAssets':'total_assets',
                       'Result_FinancialStatement NetAssets':'net_assets','IssuedShareEquityQuote IssuedShare':'num_share',
                       'EndOfDayQuote ExchangeOfficialClose':'close_official',
                       'Forecast_Dividend AnnualDividendPerShare':'dividend_annual_yen_forecast',
                       'Forecast_FinancialStatement ReportType':'quater_forecast'},inplace=True)
    
    
    #配当について
    div = b[['ymd','code','dividend_annual_yen_forecast']].dropna()
    div = pd.merge(div,a[['ymd','code','cum_adjust']],on=["ymd","code"],how="left")
    div = div >> dp.group_by(dp.X.code) >> \
    dp.mutate(dl=dp.lag(dp.X.dividend_annual_yen_forecast),cal=dp.lag(dp.X.cum_adjust)) >> dp.ungroup() >> \
    dp.mutate(div_c=(dp.X.dividend_annual_yen_forecast)/(dp.X.dl*(dp.X.cum_adjust/dp.X.cal))-1)
    
    #ROE and B/P
    b = b >> dp.mutate(roe=dp.X.net_income/dp.X.net_assets,\
                       bp=(dp.X.net_assets*1000000)/(dp.X.num_share*dp.X.close_official))
    
    #銘柄ごとに前年同期の内容を取得
    #前年同期比変化率を取得(絶対値)
    qd = b >> dp.group_by(dp.X.code,dp.X.quarter) >> \
    dp.mutate(netsales_o=dp.lag(dp.X.netsales),opm_o=dp.lag(dp.X.opm),\
              orm_o=dp.lag(dp.X.orm),net_income_o=dp.lag(dp.X.net_income)) >>\
    dp.ungroup() >> dp.arrange(dp.X.ymd,dp.X.code)
    qd['netsales_qoq'] = (qd.netsales-qd.netsales_o)/abs(qd.netsales_o)
    qd['opm_qoq'] = (qd.opm-qd.opm_o)/abs(qd.opm_o)
    qd['orm_qoq'] = (qd.orm-qd.orm_o)/abs(qd.orm_o)
    qd['net_income_qoq'] = (qd.net_income-qd.net_income_o)/abs(qd.net_income_o)
    qd = qd[['ymd','code','netsales_qoq','opm_qoq','orm_qoq','net_income_qoq','accounting_standard']]
    
    
    
    
    
    #前回会社予想
    
    le = b[["ymd","code","fy","quarter","modify_ymd","netsales","opm","orm","net_income"]] >> \
    dp.distinct(dp.X.code,dp.X.fy,keep='first')
    
    ri = b[["code","fy_forecast","netsales_forecast","opm_forecast","orm_forecast","net_income_forecast"]] >> \
    dp.distinct(dp.X.code,dp.X.fy_forecast,keep='last')
    
    dif = pd.merge(le,ri,left_on=["code","fy"],right_on=["code","fy_forecast"],how="left") >> \
    dp.arrange(dp.X.code,dp.X.ymd)
    
    dif['netsales_c'] = (dif.netsales-dif.netsales_forecast)/abs(dif.netsales_forecast)
    dif['opm_c'] = (dif.opm-dif.opm_forecast)/abs(dif.opm_forecast)
    dif['orm_c'] = (dif.orm-dif.orm_forecast)/abs(dif.orm_forecast)
    dif['net_income_c'] = (dif.net_income-dif.net_income_forecast)/abs(dif.net_income_forecast)
    dif = dif[['ymd','code','netsales_c','opm_c','orm_c','net_income_c']]
    
    
    #来期予想
    g = b >> dp.group_by(dp.X.code,dp.X.quater_forecast) >> dp.mutate(\
                        rns=(dp.X.netsales_forecast-dp.lag(dp.X.netsales_forecast))/abs(dp.lag(dp.X.netsales_forecast)),\
                         rop=(dp.X.opm_forecast-dp.lag(dp.X.opm_forecast))/abs(dp.lag(dp.X.opm_forecast)),\
                         ror=(dp.X.orm_forecast-dp.lag(dp.X.orm_forecast))/abs(dp.lag(dp.X.orm_forecast)),\
                         rni=(dp.X.net_income_forecast-dp.lag(dp.X.net_income_forecast))/abs(dp.lag(dp.X.net_income_forecast))) \
                        >> dp.ungroup()
    
    g = g[["ymd","code","rns","rop","ror","rni"]]
    
    g.rns = dp.if_else(g.rns>1,1,dp.if_else(g.rns< -1,-1,g.rns))
    g.rop = dp.if_else(g.rop>1,1,dp.if_else(g.rop< -1,-1,g.rop))
    g.ror = dp.if_else(g.ror>1,1,dp.if_else(g.ror< -1,-1,g.ror))
    g.rni = dp.if_else(g.rni>1,1,dp.if_else(g.rni< -1,-1,g.rni))
    
    #g = g.fillna(0)
    
    
    #前年同期比と会社予想を結合
    funda = pd.merge(b[['ymd','code','fy','quarter','modify_ymd','roe','bp']],qd,\
                     on=["ymd","code"],how="inner")
    funda = pd.merge(funda,dif,on=["ymd","code"],how="left")
    
    funda = funda.set_index(pd.to_datetime(funda.ymd))
    
    funda.ymd = funda.ymd.str.replace('/','-')
    
    d = pd.merge(funda,datas,on=["ymd","code"],how="left")
    
    d = pd.merge(d,div[["ymd","code","div_c"]],on=["ymd","code"],how="left")
    
    d = pd.merge(d,g,on=["ymd","code"],how="left")
    
    d = d >> dp.arrange(dp.X.ymd,dp.X.code)
    
    d = d.set_index(pd.to_datetime(d.ymd))
    
    d = d >> dp.select(['code','sig','s25d','s75d','cap','atv','roe','bp','netsales_qoq','opm_qoq','orm_qoq','net_income_qoq',\
                        'netsales_c','opm_c','orm_c','net_income_c','div_c','section','fy','quarter','high_20',\
                        'low_20','RVOL','RVOL60','high20_s','low20_s','RVOL_20',"rns","rop","ror","rni"])
    
    #訂正は学習に含めない
    if(skip==True):
        d = d >> dp.distinct(dp.X.code,dp.X.fy,dp.X.quarter,keep='first') >> dp.drop(dp.X.fy)
    else:
        d = d >> dp.drop(dp.X.fy)
    
    d = pd.concat([d,pd.get_dummies(d.section)],axis=1)
    d = pd.concat([d,pd.get_dummies(d.quarter)],axis=1)
    
    d.netsales_qoq = dp.if_else(d.netsales_qoq>1,1,dp.if_else(d.netsales_qoq< -1,-1,d.netsales_qoq))
    d.opm_qoq = dp.if_else(d.opm_qoq>1,1,dp.if_else(d.opm_qoq< -1,-1,d.opm_qoq))
    d.orm_qoq = dp.if_else(d.orm_qoq>1,1,dp.if_else(d.orm_qoq< -1,-1,d.orm_qoq))
    d.net_income_qoq = dp.if_else(d.net_income_qoq>1,1,dp.if_else(d.net_income_qoq< -1,-1,d.net_income_qoq))
    
    d.netsales_c = dp.if_else(d.netsales_c>1,1,dp.if_else(d.netsales_c< -1,-1,d.netsales_c))
    d.opm_c = dp.if_else(d.opm_c>1,1,dp.if_else(d.opm_c< -1,-1,d.opm_c))
    d.orm_c = dp.if_else(d.orm_c>1,1,dp.if_else(d.orm_c< -1,-1,d.orm_c))
    d.net_income_c = dp.if_else(d.net_income_c>1,1,dp.if_else(d.net_income_c< -1,-1,d.net_income_c))
    d.div_c = dp.if_else(d.div_c>1,1,d.div_c)
    
    d = d.replace([np.inf,-np.inf],np.nan)
    dataset_ = d.fillna(0)
    dataset_ = dataset_[st:ed]
    
    if(test_flag==True):
        dataset_ = dataset_[(dataset_.index.month == 3)|(dataset_.index.month == 4)|(dataset_.index.month == 5)]
    
    if(comp==True):
        e = pd.merge(dfs["stock_fin"],dfs["stock_list"],on="Local Code",how="inner") >> \
        dp.filter_by(dp.X['prediction_target']==True) >> dp.select(dp.X['base_date'],dp.X['Local Code'])
        
        e = e.set_index(pd.to_datetime(e.base_date))
        
        e.base_date = e.base_date.str.replace('/','-')
        
        e = e[st:ed]
        e = e.reset_index(drop=True) 
        e = e.rename(columns={'base_date':'ymd','Local Code':'code'})
        e.ymd = pd.to_datetime(e.ymd)
        
        dataset_ = dataset_.reset_index()
        dataset_ = pd.merge(e,dataset_,on=['ymd',"code"],how="left") >> dp.arrange(dp.X.ymd,dp.X.code)
        dataset_ = dataset_.fillna(0)
        dataset_ = dataset_.set_index(pd.to_datetime(dataset_.ymd)) >> dp.drop(dp.X.ymd)

    return dataset_
    

#高値ラベルを訓練する関数
def train_high(feat):

    train_all = feat
    train_all_ = train_all.reset_index(drop=True) 
    tr = train_all_.sample(round(len(train_all_)*0.7))
    val = train_all_.drop(tr.index)
    #high----------------------------------------
    tr_x_h = tr >> dp.drop(dp.X.code,dp.X.high_20,dp.X.low_20,dp.X.section,dp.X.quarter,dp.X.high20_s,dp.X.low20_s,dp.X.RVOL_20)
    tr_y_h = tr >> dp.select(dp.X.high20_s)
    tr_x_h = tr_x_h.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    tr_y_h = tr_y_h.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    val_x_h = val >> dp.drop(dp.X.code,dp.X.high_20,dp.X.low_20,dp.X.section,dp.X.quarter,dp.X.high20_s,dp.X.low20_s,dp.X.RVOL_20)
    val_y_h= val >> dp.select(dp.X.high20_s)
    val_x_h = val_x_h.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    val_y_h = val_y_h.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    
    #CV
    lgb_model = lgb.LGBMRegressor()
    grid_param ={'n_estimators':[3000],'max_depth':[4,8,12,16],'learning_rate':[0.1,0.05,0.01],
                 'min_data_in_bin':[10,20,30,100]}
    
    fit_params={'early_stopping_rounds':100, 
                'eval_metric' : 'rmse', 
                'eval_set' : [[val_x_h,val_y_h]]
               }
    
    lgb_cv = GridSearchCV(
                lgb_model, # 識別器
                grid_param, # 最適化したいパラメータセット 
                fit_params=fit_params,
                cv = 3, # 交差検定の回数
                scoring = 'neg_mean_squared_error',
                verbose=0)
    
    lgb_cv.fit(
                tr_x_h, 
                tr_y_h
                )
    imp = pd.DataFrame(lgb_cv.best_estimator_.feature_importances_,
                       columns=['importance'],index=lgb_cv.best_estimator_.feature_name_)
    
    
    res_dict = {"model_high":lgb_cv.best_estimator_,"importance":imp,"CV_res":lgb_cv.cv_results_}
    
    return(res_dict)

#安値ラベルを訓練する関数
def train_low(feat):
    
    train_all = feat
    train_all_ = train_all.reset_index(drop=True) 
    tr = train_all_.sample(round(len(train_all_)*0.7))
    val = train_all_.drop(tr.index)
    
    tr_x_l = tr >> dp.drop(dp.X.code,dp.X.high_20,dp.X.low_20,dp.X.section,dp.X.quarter,dp.X.high20_s,dp.X.low20_s,dp.X.RVOL_20)
    tr_y_l = tr >> dp.select(dp.X.low20_s)
    tr_x_l = tr_x_l.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    tr_y_l = tr_y_l.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    val_x_l = val >> dp.drop(dp.X.code,dp.X.high_20,dp.X.low_20,dp.X.section,dp.X.quarter,dp.X.high20_s,dp.X.low20_s,dp.X.RVOL_20)
    val_y_l = val >> dp.select(dp.X.low20_s)
    val_x_l = val_x_l.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    val_y_l = val_y_l.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    
    #CV
    lgb_model = lgb.LGBMRegressor()
    grid_param ={'n_estimators':[3000],'max_depth':[4,8,12,16],'learning_rate':[0.1,0.05,0.01],
                 'min_data_in_bin':[10,20,30,100]}
    
    fit_params={'early_stopping_rounds':100, 
                'eval_metric' : 'rmse', 
                'eval_set' : [[val_x_l,val_y_l]]
               }
    
    lgb_cv = GridSearchCV(
                lgb_model, # 識別器
                grid_param, # 最適化したいパラメータセット 
                fit_params=fit_params,
                cv = 3, # 交差検定の回数
                scoring = 'neg_mean_squared_error',
                verbose=0)
    
    lgb_cv.fit(
                tr_x_l, 
                tr_y_l
                )
    
    imp = pd.DataFrame(lgb_cv.best_estimator_.feature_importances_,
                       columns=['importance'],index=lgb_cv.best_estimator_.feature_name_)
    
    
    res_dict = {"model_low":lgb_cv.best_estimator_,"importance":imp,"CV_res":lgb_cv.cv_results_}
    
    return(res_dict)
    
#入力されたモデルでtestsetを予測
def predict(model,testset):

    testset = testset >> \
    dp.drop(dp.X.code,dp.X.high_20,dp.X.low_20,dp.X.section,dp.X.quarter,dp.X.high20_s,dp.X.low20_s,dp.X.RVOL_20)
    
    return(pd.Series(model.predict(testset)))
    


#インプットデータが保存されているディレクトリのパスを入力
DATASET_DIR = ""
#インプットデータを取得
data = get_dataset(inputs=get_inputs(dataset_dir=DATASET_DIR))
#学習データを期間を指定して取得
#サブミットした学習済みモデルは20170101-20201231のデータで学習しています
train = get_feat(dfs=data,st="2017-01-01",ed="2020-12-31",skip=True,test_flag=False,comp=False)
#テストデータを期間を指定して取得
test = get_feat(dfs=data,st="2020-01-01",ed="2020-12-31",skip=False,test_flag=False,comp=False)




#高値ラベルを予測するモデルを訓練
res_high = train_high(feat=train)
#特徴量重要度
res_high["importance"]
#フィッテイングの結果を表示
res_high["CV_res"]
#学習済みモデルを取り出す
model_high = res_high["model_high"]


#安値ラベルを予測するモデルを訓練
res_low = train_low(feat=train)
#特徴量重要度
res_low["importance"]
#フィッテイングの結果を表示
res_low["CV_res"]
#学習済みモデルを取り出す
model_low = res_low["model_low"]

#テストデータにおける高値を予測
pred_high = pd.DataFrame({'pred':predict(model=model_high,testset=test)})
pred_high['high'] = pd.Series(test.high_20).reset_index().high_20
#正解ラベルとの順位相関を計算
pred_high.corr(method="spearman")

#テストデータにおける安値を予測
pred_low = pd.DataFrame({'pred':predict(model=model_low,testset=test)})
pred_low['low'] = pd.Series(test.low_20).reset_index().low_20
#正解ラベルとの順位相関を計算
pred_low.corr(method="spearman")










