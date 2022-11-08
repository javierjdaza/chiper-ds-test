import streamlit as st
import pandas as pd


# hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.set_page_config(page_title='Chiper Test',layout='wide',page_icon="./img/ch.png")
col1_,col2_,col3_ = st.columns(3)
with col1_:
    st.image('./img/chiper_logo.png')
st.write('---') 
st.markdown('## Code 📗')
st.write('---') 


st.markdown('### Importing Libraries')
st.code("""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

""")


st.markdown('### Read File')
st.code("""
df = pd.read_excel('./data/archivo.xlsx')
""")

st.markdown('### Data Wrangling & Feature Engineering')
st.code("""
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
df.dtypes""")


st.markdown('### AUX Fuctions for Data Visualization')
st.code("""
def line_plot_hue(df:pd.DataFrame, x_feature:str, y_feature:str, hue: str):
    red_blue = ['#EF4836','#19B5FE']

    fig,ax = plt.subplots(figsize=(20, 7))


    plt.title(f'{x_feature} vs {y_feature} \n Line Plot', fontsize=14)
    data = df.groupby([f'{x_feature}',f'{hue}'])[f'{y_feature}'].sum().reset_index()
    g = sns.lineplot(data=data, x=f'{x_feature}', y=f'{y_feature}', style=f'{hue}', hue = f'{hue}', markers=True, dashes=False,ci=None, palette=red_blue )
    g.axhline(data[y_feature].mean(), color = 'black', lw = 1, ls='--')  
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel(f'{x_feature}', fontsize=14)
    plt.ylabel(f'{y_feature}', fontsize=14)
    # plt.ylim(df[y_feature].min(), df[y_feature].max())
    plt.show();


def plt_hist_with_hue(target_one:pd.DataFrame, target_cero: pd.DataFrame, feature:str, bins = 30, normalize = True)->plt:
   
    red_blue = ['#EF4836','#19B5FE']
    palette = sns.color_palette(red_blue)
    sns.set_palette(palette)
    sns.set_style("white")

    fig,ax = plt.subplots(figsize=(20, 7))

    #   {feature}
    plt.title(f'{feature}Distribution', fontsize=16)
    target_one[f'{feature}'].hist( alpha=0.7, bins=bins, label = 'January',density=normalize)
    target_cero[f'{feature}'].hist( alpha=0.7, bins=bins, label = 'February',density=normalize)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel(f'{feature}', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    # fig.tight_layout()
    plt.show()

""")

st.markdown('### Group Data by `day_of_year` for trainning Model')
st.code("""
x = df.groupby(['day_of_year','unit_price_USD','category','macroCategory','week_of_year','day_of_month','month_number','stock','sku'])['totalSales'].sum().reset_index()
""")

st.markdown('### Splitting Data & Generate Pipeline')
st.code("""
X = x.drop(columns = {'totalSales'}, axis = 1)
y = x['totalSales']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5, random_state = 0)

categorical_columns = X_train.select_dtypes(include=['object']).columns.to_list()
numerical_columns = X_train.select_dtypes(exclude=['object']).columns.to_list()

column_transformers = ColumnTransformer([
('MinMaxScaler', MinMaxScaler(),numerical_columns),
('One Hot Encoding', OneHotEncoder(sparse=False, handle_unknown= 'ignore'),categorical_columns)
])


model = RandomForestRegressor()
model_name = model.__class__.__name__       


# Build the pipeline
pipe = Pipeline([
                    ('Column Transformer',column_transformers),
                    ('Model', model)
                ])

""")

st.markdown('### Fit & Predict')
st.code("""
pipe.fit(X_train, y_train);
y_predict = pipe.predict(X_test)
print(f'mae = {mean_absolute_error(y_test, y_predict)}')
print(f'r2 = {r2_score(y_test, y_predict)}')
""")



st.markdown('### Export Model (.pkl File)')
st.code("""
import pickle
import os

model_save_path = './model'
output_file = os.path.join(model_save_path, 'model.pkl')
pickle.dump(pipe,open(output_file,'wb'))
""")


st.markdown('### Generate Dataframe with `y_predict`')
st.code("""
y = pd.DataFrame(y_test)
y['y_predict'] = [ int(i) for i in y_predict]
try:
    y['day_of_year'] = X_test
except:
    y = y.join(X_test)
""")
st.write('---')
st.markdown('## Plots 🧩')
st.write('---')
st.markdown('### Correlation Plot')
st.code("""
sns.heatmap(x.corr(), annot=True)
""")


st.markdown('### Generate line plot to compare `y_test` & `y_predict`')
st.code("""
red_blue = ['#EF4836','#19B5FE']
sns.lineplot(data = y, x = 'day_of_year', y = 'totalSales', markers=True, dashes=False,ci=None, label = 'y_true')
sns.lineplot(data = y, x = 'day_of_year', y = 'y_predict',markers=True, dashes=False,ci=None, label = 'y_predict')
plt.show();
""")



st.markdown('### Feaute Importance Plot')
st.code("""
def plot_feature_importance_from_pipeline(pipe: Pipeline, top_n_features:int, model_name = model_name):
    colums_after_pipelines = pipe[:-1].get_feature_names_out()
    feature_importances_list = pipe.named_steps['Model'].feature_importances_
    zipped = zip(list(feature_importances_list), list(colums_after_pipelines))
    feature_importances_df = pd.DataFrame(zipped, columns = ['importances','feature'])
    feature_importances_df.sort_values(by = 'importances', inplace = True, ascending=False)
    feature_importances_df = feature_importances_df.head(top_n_features)
    feature_importances_df['feature'] = feature_importances_df['feature'].apply(lambda x: x.replace('MinMaxScaler__',''))

    plt.title('Feature Importance\ntop10 Features')
    plt.barh( feature_importances_df['feature'],feature_importances_df['importances'], )
    plt.xlabel(f"{model_name}")


    return feature_importances_df

feature_importances_df = plot_feature_importance_from_pipeline(pipe,top_n_features=10)
""")


st.write('---')


with open("./chiper.ipynb", "rb") as file:
    btn = st.download_button(
            label="⏬ Download the Notebook File ⏬",
            data=file,
            file_name="chiper_notebook.ipynb",

          )