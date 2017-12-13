from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,
    Independence,Autoregressive)
from statsmodels.genmod.families import Poisson
from scipy.stats import skellam
import xgboost as xgb
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline

# code for debugging
# make sure to comment this out when loading from EPL_Master
# import pickle
#
# with open('tmp.pkl','rb') as f:
#     obj = pickle.load(f)
#
# hist_data = obj['hist_data']
# feature_list = obj['features_list']


# index = np.linspace(start=0,stop=(len(hist_data)-1), num=(len(hist_data)))
# index = list(index.astype(int))
# np.random.shuffle(index)
#
# train_index = index[:int(np.round(len(index)*0.7))]
# test_index = index[int(np.round(len(index)*0.7)):]

#np.array(hist_data["team_1_score"])[train_index]
### build xgboost model ###

# dtrain = xgb.DMatrix(data=np.matrix(hist_data[feature_list])[train_index,:]
#                     ,label=np.array(hist_data["team_2_score"])[train_index]
#                     , feature_names=feature_list)
#
# dtest = xgb.DMatrix(data=np.matrix(hist_data[feature_list])[test_index,:]
#                     ,label=np.array(hist_data["team_2_score"])[test_index]
#                     , feature_names=feature_list)

def BuildPoissonXGBTree(hist_data, feature_list, comp_data = None):
    ''' Build score predictions via (tree based) poisson regression. '''
    
    dtrain_1 = xgb.DMatrix(data=np.matrix(hist_data[feature_list])
                         ,label=np.array(hist_data["team_1_score"])
                         , feature_names=feature_list)

    dtrain_2 = xgb.DMatrix(data=np.matrix(hist_data[feature_list])
                         ,label=np.array(hist_data["team_2_score"])
                         , feature_names=feature_list)

    param_1 = {'max_depth': 2, 'eta': 0.1, 'silent': 1, 'objective': 'count:poisson'}
    param_1['nthread'] = 8
    param_1['eval_metric'] = 'poisson-nloglik'

    param_2 = {'max_depth': 2, 'eta': 0.1, 'silent': 1, 'objective': 'count:poisson'}
    param_2['nthread'] = 8
    param_2['eval_metric'] = 'poisson-nloglik'



    #evallist_1 = [(dtrain, 'train'),(dtest, 'test')]
    evallist_1 = [(dtrain_1, 'train')]

    #evallist_2 = [(dtrain, 'train'),(dtest, 'test')]
    evallist_2 = [(dtrain_2, 'train')]

    num_round = 100
    bst_1 = xgb.train(param_1, dtrain_1, num_round, evallist_1)
    bst_2 = xgb.train(param_2, dtrain_2, num_round, evallist_2)

    ypred_1 = bst_1.predict(dtrain_1)
    ypred_2 = bst_2.predict(dtrain_2)


    hist_data["team_1_score_pred"] = ypred_1
    hist_data["team_2_score_pred"] = ypred_2

    #hist_data[['team_1_score','team_1_score_pred','team_2_score','team_2_score_pred']]
    if comp_data is None:
        return hist_data

    dcomp = xgb.DMatrix(data=np.matrix(comp_data[feature_list])
                         , feature_names=feature_list)

    # prepare comp data
    comp_data['team_1_score_pred'] = bst_1.predict(dcomp)
    comp_data['team_2_score_pred'] = bst_2.predict(dcomp)

    comp_data['team_1_prob'] = comp_data[['team_1_score_pred','team_2_score_pred']].apply(lambda x: 1 - skellam.cdf(0,x['team_1_score_pred'],x['team_2_score_pred']), 1)
    comp_data['team_tie_prob'] = comp_data[['team_1_score_pred','team_2_score_pred']].apply(lambda x: skellam.pmf(0,x['team_1_score_pred'],x['team_2_score_pred']), 1)
    comp_data['team_2_prob'] = comp_data[['team_1_score_pred','team_2_score_pred']].apply(lambda x: skellam.cdf(-1,x['team_1_score_pred'],x['team_2_score_pred']), 1)

    return hist_data, comp_data






# def plot_importance(booster, figsize):
#     from matplotlib import pyplot as plt
#     from xgboost import plot_importance
#     fig, ax = plt.subplots(1,1,figsize=figsize)
#     return xgb.plot_importance(booster=booster, ax=ax)

#plot_importance(bst_2,(14, 11))

def BuildPoissonModels(hist_data, feature_list, comp_data = None):
    ''' Build score predictions via (linear) poisson regression. '''
    hist_data_1 = hist_data[["team_1_score"] + feature_list]
    hist_data_2 = hist_data[["team_2_score"] + feature_list]

    formula_1 = "team_1_score ~ " + " + ".join(feature_list)
    formula_2 = "team_2_score ~ " + " + ".join(feature_list)

    # using the GEE package along with independance assumptions to fit poisson model.
    # Am assuming this is using a maximum likleyhood approach?
    fam = Poisson()
    ind = Independence()


    model_1 = GEE.from_formula(formula_1, "team_1_score", hist_data, cov_struct=ind, family=fam)
    model_2 = GEE.from_formula(formula_2, "team_2_score", hist_data, cov_struct=ind, family=fam)

    model_1_fit = model_1.fit()
    model_2_fit = model_2.fit()
    print(model_1_fit.summary())


    hist_data['team_1_score_pred'] = model_1_fit.predict(hist_data)
    hist_data['team_2_score_pred'] = model_2_fit.predict(hist_data)

    # return historical data if comp_data wasn't passed.
    if comp_data is None:
        return hist_data

    # prepare comp data
    comp_data['team_1_score_pred'] = model_1_fit.predict(comp_data[feature_list])
    comp_data['team_2_score_pred'] = model_2_fit.predict(comp_data[feature_list])

    comp_data['team_1_prob'] = comp_data[['team_1_score_pred','team_2_score_pred']].apply(lambda x: 1 - skellam.cdf(0,x['team_1_score_pred'],x['team_2_score_pred']), 1)
    comp_data['team_tie_prob'] = comp_data[['team_1_score_pred','team_2_score_pred']].apply(lambda x: skellam.pmf(0,x['team_1_score_pred'],x['team_2_score_pred']), 1)
    comp_data['team_2_prob'] = comp_data[['team_1_score_pred','team_2_score_pred']].apply(lambda x: skellam.cdf(-1,x['team_1_score_pred'],x['team_2_score_pred']), 1)

    return hist_data, comp_data



def PoissonNLogLiklihood(df, score_name, pred_name):
    ''' calculates the negative log liklihood of a dataframe of predictions '''
    return(
        df[[score_name,pred_name]].\
            apply(lambda x: -1*(x[score_name] * np.log(x[pred_name])
                                - x[pred_name]
                                - np.log(np.math.factorial(x[score_name])))
                 , 1 ).mean()
            )
