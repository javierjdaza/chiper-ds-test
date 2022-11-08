from calendar import month
from datetime import timedelta
import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,timedelta


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
    return fig

def plt_hist_with_hue(target_one:pd.DataFrame, target_cero: pd.DataFrame, feature:str, bins = 30, normalize = True)->plt:
    """Plot 2 histograms in 1 plot, with alpha 

    Args:
        target_one (pd.DataFrame): dataframe with target value = 1
        target_cero (pd.DataFrame): dataframe with target value = 0
        feature (str): feature name 

    Returns:
        plt: matplot lib chart
    """
    red_blue = ['#EF4836','#19B5FE']
    palette = sns.color_palette(red_blue)
    sns.set_palette(palette)
    sns.set_style("white")

    fig,ax = plt.subplots(figsize=(20, 7))

    #   {feature}
    plt.title(f'{feature}\nDistribution', fontsize=16)
    target_one[f'{feature}'].hist( alpha=0.7, bins=bins, label = 'January',density=normalize)
    target_cero[f'{feature}'].hist( alpha=0.7, bins=bins, label = 'February',density=normalize)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.xlabel(f'{feature}', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    # fig.tight_layout()
    plt.show()
    return fig

st.set_page_config(page_title='Chiper Test',layout='wide',page_icon="./img/ch.png")
col1_,col2_,col3_ = st.columns(3)
with col1_:
    st.image('./img/chiper_logo.png')
st.write('---') 
st.markdown('## Correlations & Plots 📊')
st.write('---') 
df = read_df()
df = feature_engineering(df)


st.markdown('### Data + Feature Engineering 🥷')
st.dataframe(df)

# -------------------
# PLOTS
# -------------------


st.write('---') 
st.markdown('### Line Plot 📈')
c1,c2,c3 = st.columns(3)
st.write(' ')
with c2:
    time_frame = st.selectbox('Select Time Frame for plot', ['week_of_year','day_of_month','day_name','day_of_year'])
st.pyplot(line_plot_hue( df = df, x_feature = time_frame, y_feature = 'totalSales', hue = 'month'))
st.caption('*the char will change according to the time frame selected.')

st.write('---') 


target_one = df[df['month'] == 'january'].copy()
target_cero = df[df['month'] == 'february'].copy()

st.markdown('### Distribution Plot 📊')
p1,p2,p3 = st.columns(3)
st.write(' ')
with p2:
    bins = st.select_slider('Select bins', options = [10,30,50,100])
st.pyplot(plt_hist_with_hue(target_one,target_cero,feature = 'totalUSD', bins = bins))
st.caption('*the char will change according to the bins frame selected.')
st.write('---') 



st.markdown('### Correlation Plot 📊')


def correlation_plot(df):
    fig,ax = plt.subplots(figsize=(20, 7))
    plt.title(f'Correlation Plot', fontsize=16)
    keep_cols = ['totalSales', 'date', 'stock', 'unit_price_USD', 'week_of_year', 'day_of_month', 'day_of_year']
    corr_df = df[keep_cols].copy()
    sns.heatmap(corr_df.corr(), annot=True)
    return fig
st.pyplot(correlation_plot(df))