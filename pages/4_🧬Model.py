from calendar import month
from datetime import timedelta
import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,timedelta
import os, pickle

from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import set_config


# -------------------
# AUX Fuctions
# -------------------

@st.cache(allow_output_mutation=True)
def read_df():
    df = pd.read_excel('./data/archivo.xlsx')
    return df

@st.cache(allow_output_mutation=True)
def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by = ['date'], inplace = True)
    df['sku'] = df['sku'].astype(str)
    # ----------------------
    # FEATURE ENGINEERING
    # ----------------------
    df['unit_price_USD'] = df['totalUSD'] / df['totalSales']
    df['unit_price_USD'] = df['unit_price_USD'].apply(lambda x: round(x,3))

    # Changing date to datetime and Creating New Columns
    df['week_of_year'] = df['date'].apply(lambda x: x.strftime('%W').lower()).astype(int)
    df['day_of_month'] = df['date'].apply(lambda x: x.strftime('%d')).astype(int)
    df['fortnite'] = df['day_of_month'].apply(lambda x: 1 if x <=15 else 2)
    df['month'] = df['date'].apply(lambda x: x.strftime('%B').lower())
    df['month_number'] = df['date'].apply(lambda x: x.strftime('%m').lower())
    df['day_name'] = df['date'].apply(lambda x: x.strftime('%A').lower())
    df['day_of_year'] = df['date'].apply(lambda x: x.strftime('%-j').lower()).astype(int)
    return df

@st.cache(allow_output_mutation=True)
def load_model():
    # Read pipeline (.pkl)
    pipe = pickle.load(open('./model/model.pkl', 'rb'))
    return pipe


def plot_feature_importance_from_pipeline(pipe: Pipeline, top_n_features:int, model_name = 'RandomForestRegressor'):
    
    colums_after_pipelines = pipe[:-1].get_feature_names_out()
    feature_importances_list = pipe.named_steps['Model'].feature_importances_
    zipped = zip(list(feature_importances_list), list(colums_after_pipelines))
    feature_importances_df = pd.DataFrame(zipped, columns = ['importances','feature'])
    feature_importances_df.sort_values(by = 'importances', inplace = True, ascending=False)
    feature_importances_df = feature_importances_df.head(top_n_features)
    feature_importances_df['feature'] = feature_importances_df['feature'].apply(lambda x: x.replace('MinMaxScaler__',''))
    fig,ax = plt.subplots(figsize=(20, 7))
    plt.title('Feature Importance\ntop10 Features',fontsize=16)
    plt.barh( feature_importances_df['feature'],feature_importances_df['importances'], )
    plt.xlabel(f"{model_name}",fontsize=14)
    return fig


def plot_results_predict(results_predict):
    fig,ax = plt.subplots(figsize=(20, 7))
    plt.title('Actual values vs Predict Values',fontsize=16 )
    sns.lineplot(data = results_predict, x = 'day_of_year', y = 'totalSales', markers=True, dashes=False,ci=None, label = 'y_true')
    sns.lineplot(data = results_predict, x = 'day_of_year', y = 'y_predict',markers=True, dashes=False,ci=None, label = 'y_predict')
    plt.xlabel(f'day_of_year', fontsize=14)
    plt.ylabel('totalSales', fontsize=14)
    plt.show();
    return fig





st.set_page_config(page_title='Chiper Test',layout='wide',page_icon="./img/ch.png")
col1_,col2_,col3_ = st.columns(3)
with col1_:
    st.image('./img/chiper_logo.png')
st.write('---') 
st.markdown('## Model 🧬')
st.write('---') 

original_df = read_df()
original_df = feature_engineering(original_df)
df = original_df.groupby(['day_of_year','unit_price_USD','category','macroCategory','week_of_year','day_of_month','month_number','stock','sku'])['totalSales'].sum().reset_index()


st.markdown('### Data + Feature Engineering 🥷')
st.dataframe(df)



# -------------------
# Model
# -------------------
X = df.drop(columns = {'totalSales'}, axis = 1)
y = df['totalSales']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5, random_state = 0)

model = load_model()
y_predict = model.predict(X_test)
st.write('---') 
st.markdown('### Pipeline Steps 🧨')
st.image('./img/pipeline_img.png')
st.write('---') 
st.markdown('### Model Metrics ⚒')


metrics = { 'model_name': 'RandomForestRegressor',
            'MAE' : round(mean_absolute_error(y_test, y_predict),2),
            'R2' : round(r2_score(y_test, y_predict),2)
}
metrics_df = pd.DataFrame([metrics])
st.dataframe(metrics_df)

results_predict = pd.DataFrame(y_test)
results_predict['y_predict'] = [ int(i) for i in y_predict]
# results_predict['error'] = results_predict['y_predict'] - y['totalSales']
try:
    results_predict['day_of_year'] = X_test
except:
    results_predict = results_predict.join(X_test)
st.markdown('##### Data with prediction `y_predict` column')
st.dataframe(results_predict)
st.write('---')




st.pyplot(plot_results_predict(results_predict))

st.write('---')


st.pyplot(plot_feature_importance_from_pipeline(model,top_n_features=10))
st.write('---')


with open("./model/model.pkl", "rb") as file:
    btn = st.download_button(
            label="⏬ Download the Pickle File 🧬 ⏬",
            data=file,
            file_name="chiper_model.pkl",

          )