import numpy as np
import pandas as pd
import os
#os.chdir('EPL')
import talib
from tqdm import tqdm
import datetime
import calendar
import peyton




### build time features.

# first fourier component (yearly periodicity)

#
# time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
#
# day_of_year = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').timetuple().tm_yday
#
# weekday = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').weekday()
# hour = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').hour
# minute = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').minute
#
#
# weekday += hour/24
#
# weekday

def year_fourier_components(time):
    day_of_year = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').timetuple().tm_yday

    if calendar.isleap(datetime.datetime.now().year):
        x = np.cos(day_of_year/366*2*np.pi)
        y = np.sin(day_of_year/366*2*np.pi)
    else:
        x = np.cos(day_of_year/365*2*np.pi)
        y = np.sin(day_of_year/365*2*np.pi)

    return(x,y)

def week_fourier_components(time):
    weekday = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').weekday()
    hour = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').hour

    weekday += hour/24

    x = np.cos(weekday/7*2*np.pi)
    y = np.sin(weekday/7*2*np.pi)

    return(x,y)

def day_fourier_components(time):
    minute = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S').minute

    x = np.cos(minute/60*2*np.pi)
    y = np.sin(minute/60*2*np.pi)

    return(x,y)

# calculate team scores
def team_result(team_score,opponent_score):
    if team_score > opponent_score:
        return('team_win')
    elif team_score < opponent_score:
        return('team_loss')
    else:
        return('tie')


# function to generate rolling match result indicies
def GenerateRollingResultIndex(index_roll, tmp_team_frame):
    # team_1_looks a lot like home team, so will want an adjusted 'weighted' score as well. Will design this to be rolling.
    #index_roll = 20

    tmp_team_frame['result'] = tmp_team_frame[['team_score','opponent_score']].apply(lambda x: team_result(x[0],x[1]), 1)
    team_type_stats = tmp_team_frame.iloc[:index_roll].pivot_table(index = 'team_type', columns = 'result', values = 'team_score' , aggfunc = len ).reset_index()

    if len(team_type_stats.columns.tolist()) < 4:
        return(tmp_team_frame.iloc[index_roll:index_roll+1])


    team_type_matrix = np.matrix(team_type_stats.sort_values('team_type')[['team_win','team_loss','tie']])

    # calcualte indeices
    team_type_result_ratios = team_type_matrix / np.repeat(team_type_matrix.sum(axis = 1),3,axis=1)
    team_type_result_indices = team_type_result_ratios / np.repeat(team_type_matrix.sum(axis = 0) / team_type_matrix.sum(axis = 0).sum(),2,axis=0)

    # turn back into dataframe and join
    tmp_result_frame = pd.DataFrame(team_type_result_indices)
    tmp_result_frame.columns = ['team_win_index','team_loss_index','tie_index']
    tmp_result_frame = pd.concat([team_type_stats.sort_values('team_type'), tmp_result_frame], axis = 1)

    tmp_result_frame = pd.merge(
                                tmp_team_frame.iloc[index_roll:index_roll+1],
                                tmp_result_frame,
                                on = 'team_type'
                                )

    return(tmp_result_frame)


# helper functions for feature building below:
#team_frame = team_frames[-1]
def build_rolling_features(team_frame):

    # remove any nan scores
    tmp_team_frame = team_frame.sort_values(['date'])

    # get match count in EPL
    tmp_team_frame = tmp_team_frame.reset_index(drop = True).reset_index()
    tmp_team_frame['match_count'] = tmp_team_frame['index']+1
    tmp_team_frame = tmp_team_frame.drop('index', axis = 1)

    # get rolling moving averages of scores (rolling offensive properties)
    tmp_team_frame['team_score_ma_10'] = talib.MA(np.array([np.nan] + np.array(tmp_team_frame['team_score'])[:-1].tolist()), timeperiod = 10, matype=0)
    tmp_team_frame['team_score_ema_10'] = talib.EMA(np.array([np.nan] + np.array(tmp_team_frame['team_score'])[:-1].tolist()), timeperiod = 10)
    tmp_team_frame['team_last_score'] = talib.MA(np.array([np.nan] + np.array(tmp_team_frame['team_score'])[:-1].tolist()), timeperiod = 1, matype=0)
    # get rolling moveing averages of scores (rolling defensive properties)
    tmp_team_frame['opponent_score_ma_10'] = talib.MA(np.array([np.nan] + np.array(tmp_team_frame['opponent_score'])[:-1].tolist()), timeperiod = 10, matype=0)
    tmp_team_frame['opponent_score_ema_10'] = talib.EMA(np.array([np.nan] + np.array(tmp_team_frame['opponent_score'])[:-1].tolist()), timeperiod = 10)
    tmp_team_frame['opponent_last_score'] = talib.MA(np.array([np.nan] + np.array(tmp_team_frame['opponent_score'])[:-1].tolist()), timeperiod = 1, matype=0)

    # generate the rolling 'match result' indices
    # note that this is really slow at the moment. Will vectorize this operation at some point.
    tmp_frames = []
    for index in range(len(tmp_team_frame)):
        tmp_frames.append(GenerateRollingResultIndex(index, tmp_team_frame))

    tmp_team_frame = pd.concat(tmp_frames)

    return(tmp_team_frame)

def win_tie_loss(team_1_score,team_2_score):
    if team_1_score > team_2_score:
        return('team_1_win')
    elif team_1_score < team_2_score:
        return('team_2_win')
    else:
        return('tie')



def GetFrameWithFeatures():

    # get competition data via peyton
    throne = peyton.Throne(username='smstojanovic', token=os.environ['THRONE_TOKEN'])

    # Get historical data for a competition
    throne.competition('English Premier League').get_historical_data()
    df = throne.competition.historical_data

    # Get competition data for a competition
    throne.competition('English Premier League').get_competition_data()
    df_comp = throne.competition.competition_data




    #df = pd.read_csv('english_premier_league_historical_data.csv')
    #df_comp = pd.read_csv('english_premier_league_competition_data.csv')

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



    #tmp_team_frame

    team_frames = []
    # break out dataframe by team (manual group by for ability to do complex feature construction)
    for team in tqdm(teams):
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
        tmp_team_frame = build_rolling_features(tmp_team_frame)

        # break out dataframe by team (manual group by for ability to do complex feature construction)
        team_frames.append(tmp_team_frame)


    df['result'] = df[['team_1_score','team_2_score']].apply(lambda x: win_tie_loss(x['team_1_score'],x['team_2_score']),1)


    team_frame = pd.concat(team_frames,axis=0)

    team_frame['result'] = team_frame[['team_score','opponent_score']].apply(lambda x: win_tie_loss(x['team_score'],x['opponent_score']),1)



    # bring all team features together
    #team_frame[['id','match_count','opponent_last_score','opponent_score_ema_10','opponent_score_ma_10','team_last_score','team_score_ema_10','team_score_ma_10','team_win_index','team_loss_index','tie_index']]

    team_frame_1 = team_frame[['id','team_name','match_count','opponent_last_score','opponent_score_ema_10','opponent_score_ma_10','team_last_score','team_score_ema_10','team_score_ma_10','team_win_index','team_loss_index','tie_index']].copy()
    team_frame_1.columns = ['team_1_' + x if index > 1 else x for index, x in enumerate(team_frame_1.columns.tolist())]

    team_frame_2 = team_frame[['id','team_name','match_count','opponent_last_score','opponent_score_ema_10','opponent_score_ma_10','team_last_score','team_score_ema_10','team_score_ma_10','team_win_index','team_loss_index','tie_index']].copy()
    team_frame_2.columns = ['team_2_' + x if index > 1 else x for index, x in enumerate(team_frame_2.columns.tolist())]

    #team_frame

    #len(df)

    df = pd.merge(
        df,
        team_frame_1,
        left_on = ['id','team_1_name'],
        right_on = ['id','team_name']
    )

    df = pd.merge(
        df,
        team_frame_2,
        left_on = ['id','team_2_name'],
        right_on = ['id','team_name']
    )



    df['x_year'] = df['date'].apply(lambda x: year_fourier_components(x)[0])
    df['y_year'] = df['date'].apply(lambda x: year_fourier_components(x)[1])
    df['x_week'] = df['date'].apply(lambda x: week_fourier_components(x)[0])
    df['y_week'] = df['date'].apply(lambda x: week_fourier_components(x)[1])
    df['x_day'] = df['date'].apply(lambda x: day_fourier_components(x)[0])
    df['y_day'] = df['date'].apply(lambda x: day_fourier_components(x)[1])



    df[['id','data_type','result','team_1_score','team_2_score','is_february','is_november','c_ability_3','d_ability_1','d_ability_3','d_ability_4', 'd_form_4', 'd_h2h_2', 'team_1_name', 'team_2_name',
        'team_1_match_count', 'team_1_opponent_last_score',
               'team_1_opponent_score_ema_10', 'team_1_opponent_score_ma_10',
               'team_1_team_last_score', 'team_1_team_score_ema_10',
               'team_1_team_score_ma_10', 'team_1_team_win_index',
               'team_1_team_loss_index', 'team_1_tie_index',
        'team_2_match_count', 'team_2_opponent_last_score',
               'team_2_opponent_score_ema_10', 'team_2_opponent_score_ma_10',
               'team_2_team_last_score', 'team_2_team_score_ema_10',
               'team_2_team_score_ma_10', 'team_2_team_win_index',
               'team_2_team_loss_index', 'team_2_tie_index',
               'x_year','y_year','x_week','y_week','x_day','y_day']].to_csv('epl_data_w_features.csv',index=False)

    return(df)
