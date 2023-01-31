import pandas as pd

df = pd.read_csv('Clean_Bengaluru_House_Data.csv')

X = df.drop('price',axis=1)
y = df['price']

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 28)

transformer = make_column_transformer((OneHotEncoder(sparse=False),['location']),remainder='passthrough')

scaler = StandardScaler()

base_elastic_model = ElasticNet()

param_grid = {'alpha':[0.1,1,5,10,50,100],
              'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}

grid_model = GridSearchCV(estimator=base_elastic_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5)

pipe = make_pipeline(transformer,scaler,grid_model)

pipe.fit(X_train,y_train)

import pickle

pickle.dump(pipe, open('model.pkl','wb'))

