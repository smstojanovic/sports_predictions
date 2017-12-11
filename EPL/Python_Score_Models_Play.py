from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,
    Independence,Autoregressive)
from statsmodels.genmod.families import Poisson
from scipy.stats import skellam
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# code for debugging
# make sure to comment this out when loading from EPL_Master
import pickle

with open('tmp.pkl','rb') as f:
    obj = pickle.load(f)

hist_data = obj['hist_data']
feature_list = obj['features_list']

np.matrix(hist_data[feature_list])

hist_data["team_1_score"]

### build xgboost model ###

dtrain = xgb.DMatrix(data=np.matrix(hist_data[feature_list])
                    ,label=hist_data["team_1_score"]
                    , feature_names=feature_list)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'count:poisson'}
param['nthread'] = 8
param['eval_metric'] = 'poisson-nloglik'

evallist = [(dtrain, 'train')]

num_round = 100
bst = xgb.train(param, dtrain, num_round, evallist)

ypred = bst.predict(dtrain)

hist_data["team_1_pred"] = ypred

hist_data[['team_1_score','team_1_pred']]



def plot_importance(booster, figsize):
    from matplotlib import pyplot as plt
    from xgboost import plot_importance
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return xgb.plot_importance(booster=booster, ax=ax)

plot_importance(bst,(14, 11))



xgb.plot_tree(bst, num_trees=2)

def BuildPoissonModels(hist_data, feature_list, comp_data = None):

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
