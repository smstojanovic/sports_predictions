import numpy as np
import pandas as pd
import os
import talib

df = pd.read_csv('english_premier_league_historical_data.csv')
df_comp = pd.read_csv('english_premier_league_competition_data.csv')

# push additional columns
df['team_1_prob'] = np.nan
df['team_2_prob'] = np.nan
df['team_tie_prob'] = np.nan
df['data_type'] = 'hist'

df_comp['team_1_score'] = np.nan
df_comp['team_2_score'] = np.nan
df_comp['data_type'] = 'comp'

# bring together dfs
df = pd.concat([df, df_comp], axis=0)

# sort by time.
df = df.sort_values(['date'])

# get team list
teams = list(set(df['team_1_name'].unique().tolist()) | set(df['team_2_name'].unique().tolist()))


# helper functions for feature building below:
tmp_team_frame = team_frames[-1]
#def build_features(team_frame):

# remove any nan scores
#tmp_team_frame = tmp_team_frame[np.isnan(tmp_team_frame['team_score']) == False]
tmp_team_frame = tmp_team_frame.sort_values(['date'])

# get match count in EPL
tmp_team_frame = tmp_team_frame.reset_index(drop = True).reset_index()
tmp_team_frame['match_count'] = tmp_team_frame['index']+1
tmp_team_frame = tmp_team_frame.drop('index', axis = 1)

np.array(tmp_team_frame['team_score'])



# get rolling moving averages of scores
tmp_team_frame['team_score_ma_10'] = talib.MA(np.array([np.nan] + np.array(tmp_team_frame['team_score'])[:-1].tolist()), timeperiod = 10, matype=0)
tmp_team_frame['team_score_ema_10'] = talib.EMA(np.array([np.nan] + np.array(tmp_team_frame['team_score'])[:-1].tolist()), timeperiod = 10)
tmp_team_frame['team_last_score'] = talib.MA(np.array([np.nan] + np.array(tmp_team_frame['team_score'])[:-1].tolist()), timeperiod = 1, matype=0)


tmp_team_frame

team_frames = []
# break out dataframe by team (manual group by for ability to do complex feature construction)
team = teams[0]
for team in teams:
    # build tmp frame for all times team_1 appears
    tmp_team_1 = df[df['team_1_name'] == team].copy()
    tmp_team_1 = tmp_team_1[['id','date','team_1_score','team_2_score','team_2_name']]
    tmp_team_1.columns = ['id','date','team_score','opponent_score','opponent_name']
    tmp_team_1['team_type'] = 'team_name_1'
    # build tmp frame for all times team_1 appears
    tmp_team_2 = df[df['team_2_name'] == team].copy()
    tmp_team_2 = tmp_team_2[['id','date','team_2_score','team_1_score','team_1_name']]
    tmp_team_2.columns = ['id','date','team_score','opponent_score','opponent_name']
    tmp_team_2['team_type'] = 'team_name_2'
    # concatenate these frames and append
    tmp_team_frame = pd.concat([tmp_team_1,tmp_team_2])
    tmp_team_frame['team_name'] = team

    # build features


    # break out dataframe by team (manual group by for ability to do complex feature construction)
    team_frames.append(tmp_team_frame)


team_frames[10]


tmp_team_frame

team_1_df = df.groupby('team_1','date')

df['team_1_ma_10'] = talib.MA(np.array(df['team_1_score']), timeperiod = 10)
df['team_2_ma_10'] = talib.MA(np.array(df['team_2_score']), timeperiod = 10)


df.head(3)
