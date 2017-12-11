import pandas as pd
import numpy as np
import base_feature_extraction
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,
    Independence,Autoregressive)
from statsmodels.genmod.families import Poisson
from scipy.stats import skellam
import os
import peyton

#pd.options.display.max_columns = 50

df = base_feature_extraction.GetFrameWithFeatures()

df_model = df.copy()
df_model['value'] = 1


# bought features
bought_features = ['c_ability_3','confidence','d_ability_1','d_ability_3','d_ability_4','d_form_4','d_h2h_2']

# modelled features
modelled_features = ['team_1_match_count', 'team_1_opponent_last_score',
       'team_1_opponent_score_ema_10', 'team_1_opponent_score_ma_10',
       'team_1_team_last_score', 'team_1_team_score_ema_10',
       'team_1_team_score_ma_10', 'team_1_team_win_index',
       'team_1_team_loss_index', 'team_1_tie_index',
'team_2_match_count', 'team_2_opponent_last_score',
       'team_2_opponent_score_ema_10', 'team_2_opponent_score_ma_10',
       'team_2_team_last_score', 'team_2_team_score_ema_10',
       'team_2_team_score_ma_10', 'team_2_team_win_index',
       'team_2_team_loss_index', 'team_2_tie_index',
       'x_year','y_year','x_week','y_week','x_day','y_day']


# encode team 1 and team 2
team_1_encoded = df_model.pivot_table(index = 'id', columns = 'team_1_name', aggfunc = np.max, values='value', fill_value = 0)
team_1_encoded.columns = ['team_1_name_' + x.replace(" ","_") for x in team_1_encoded.columns.tolist()]
team_1_encoded.reset_index(inplace=True)

team_2_encoded = df_model.pivot_table(index = 'id', columns = 'team_2_name', aggfunc = np.max, values='value', fill_value = 0)
team_2_encoded.columns = ['team_2_name_' + x.replace(" ","_") for x in team_2_encoded.columns.tolist()]
team_2_encoded.reset_index(inplace=True)

df_model = df_model.drop('value', axis = 1)


df_model = pd.merge(df_model, team_1_encoded, on='id')
df_model = pd.merge(df_model, team_2_encoded, on='id')


# prepare Poisson Regression Model
hist_data = df_model[df_model['data_type'] == 'hist']
hist_data.head(3)

feature_list = modelled_features + [x for x in team_1_encoded.columns.tolist() if x != 'id'] + [x for x in team_2_encoded.columns.tolist() if x != 'id']

hist_data = hist_data[["team_1_score","team_2_score","id"] + feature_list].dropna(axis = 0)



for col in hist_data.columns.tolist():
    if col != 'id':
        hist_data[col] = hist_data[col].apply(float,1)

# remove columns that don't contribute to matrix rank
# slow way to do this. Will implement a matrix solver to automatically reduce these.
for col in feature_list:
    rank = np.linalg.matrix_rank(hist_data[feature_list])
    rank_removed = np.linalg.matrix_rank(hist_data[[x for x in feature_list if x != col]])

    if rank == rank_removed:
        feature_list = [x for x in feature_list if x != col]
        print(col)

# define competition data
comp_data = df_model[df_model['data_type'] == 'comp']

# Build Poisson Models.
import importlib
import Python_Score_Models_Play
Python_Score_Models_Play = importlib.reload(Python_Score_Models_Play)

hist_data, comp_data = Python_Score_Models_Play.BuildPoissonModels(hist_data, feature_list, comp_data)

comp_data

comp_data["confidence"] = 1.0

## subbmit comp data
# check competition results
submit_frame = comp_data[['id','team_1_name','team_2_name','team_1_prob','confidence','team_tie_prob','team_2_prob']]

submit_frame

# manual confidence lever settings.
#submit_frame.set_value(5219,'confidence', 0.5)
#submit_frame.set_value(5221,'confidence', 0.25)

#submit
throne = peyton.Throne(username='smstojanovic', token=os.environ['THRONE_TOKEN'])
throne.competition('English Premier League').submit(submit_frame)
