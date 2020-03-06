import numpy as np
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGeneralizedLinearEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2OXGBoostEstimator
h2o.init()
from graph_models.node_embedding import DeepWalk, Node2Vec
from utils import loss_function, load_data
import pandas

train_hosts, test_hosts, y_train, G = load_data('../data')

n_features = 256
n_walks = 150
walk_length = 100
embedder = DeepWalk(walk_length, n_walks, n_features, load_path='graph_models/node_embedding/models/deepwalk.model')

in_degrees = []
out_degrees = []
for node in train_hosts:
    in_degrees.append(G.in_degree(node))
    out_degrees.append(G.out_degree(node))

X_train = h2o.H2OFrame(np.concatenate((embedder.transform(train_hosts).T,[y_train])).T)



params_rf = {
    'max_depth': 20,
    'nfolds': 5,
    'ntrees': 200,
    'stopping_tolerance': 1e-2,
    'balance_classes': True,
    # 'weights_column': X_train.columns[-2],
}

params_gb = {
    'nfolds': 5,
    'learn_rate': 0.23,
    'max_depth': 1,
    'min_rows': 50,
    'ntrees': 100,
    'huber_alpha': 0.9,
    'quantile_alpha': 0.5,
    'balance_classes': True,
#    'weights_column': X_train.columns[-2],
}

params_xgb = {
    'nfolds': 5,
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth': 3,
    'min_rows': 6,
    'min_data_in_leaf': 12.,
    'ntrees': 100,
    'reg_alpha': 1.1,
    'reg_lambda': 1.1,
#    'weights_column': X_train.columns[-2],
}

params_linear = {
    'nfolds': 5,
    'family': 'multinomial',
    'lambda_search': True,
    # 'lambda_': 1e-2,
    'nlambdas': 5,
#    'alpha': 0.38
#    'alpha': 0.001,
    # 'balance_classes': False,
#    'lambda_':5e-2,
    'weights_column': X_train.columns[-2],
}

hyperparams = {
    #'lambda_': [1e-1, 1e-2, 1e-3],
    'alpha': [0, 0.25, 0.5, 0.75, 1],
    # 'solver': ["irlsm", "l_bfgs", "coordinate_descent_naive", "coordinate_descent", "gradient_descent_lh", "gradient_descent_sqerr"],
    # 'balance_classes': [True, False],
    #'weights_column': ['', X_train.columns[-2]],
}
# rf = H2ORandomForestEstimator(**params_rf)
# gb = H2OGradientBoostingEstimator(**params_gb)
xgb = H2OXGBoostEstimator(**params_xgb)
# linear = H2OGeneralizedLinearEstimator(**params_linear)
# time_per_model = 300
# aml = H2OAutoML(max_runtime_secs_per_model=time_per_model, sort_metric='logloss', balance_classes=True)
# gs = H2OGridSearch(linear, hyperparams)
# X_train, X_valid = X_train.split_frame(ratios=[0.8])

# rf.train(x=X_train.columns[:-2], y=X_train.columns[-1], training_frame=X_train, validation_frame=X_valid)
# gb.train(x=X_train.columns[:-2], y=X_train.columns[-1], training_frame=X_train)
xgb.train(x=X_train.columns[:-2], y=X_train.columns[-1], training_frame=X_train)#, validation_frame=X_valid)
# gs.train(x=X_train.columns[:-2], y=X_train.columns[-1], training_frame=X_train, validation_frame=X_valid)
# aml.train(x=X_train.columns[:-1], y=X_train.columns[-1], training_frame=X_train)
# print(rf.logloss(train=True, valid=True))
# print(gb.logloss(train=True, xval=True))
print(xgb.logloss(train=True, xval=True))
#print(linear.logloss(train=True, valid=True))
# gs.show()
# aml.download_mojo('./graph_models/best_autoML'+str(time_per_model)+'secs_n2v.MOJO')
# print(aml.leaderboard)

# model = h2o.import_mojo('./graph_models/best_autoML300secs.MOJO')
# preds = xgb.predict(h2o.H2OFrame(embedder.transform(test_hosts)))
# preds = preds.as_data_frame()

# preds.rename(columns={'predict':'Host'})
# preds['Host'] = test_hosts
# print(preds)
# preds = pandas.read_csv('../xgb_h2o.csv')

# print(preds)

# preds = preds[preds.columns.tolist()[0:1] + preds.columns.tolist()[2:]]

# print(preds)

# preds.to_csv('../xgb_h2o.csv', index=False)
