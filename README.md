# chiper-ds-test


#### Datasets Problems:
1. The problem mention Data Product Types, but the dataset aint have a column or a flag that indicates de product Type, so, include the `leadtime` in the analysis is useless
2. The data set got 2.029 diferents `sku`, thats mean 2.029 different products, with just 2 months of information, the challenge require **forecast** *(demand/sales)* the first fortnite of march by product, the amount of data is smaller than expected for make a serious prediction.

#### Feature Engineering:
1. I built new features `['week_of_year', 'day_of_month', 'fortnite', 'month', 'month_number','day_name', 'day_of_year']` using just the column `date`.
2. also i built a new feature called `unit_price_USD`, using the columns: `totalSales` & `totalUSD`

#### Features to use:
1. I dropped the following columns: `['storeReference', 'warehouseId','salesDuringCommercialActivities','locationId']`, i aint get value for them.
2. why `salesDuringCommercialActivities`?: This feature got important information, BUT, this information will be known AFTER the activities, so, for make prediction, the model will be have data leakeage, is very dangerous use a column with information known after a certain process, like this column.
3. I would not have used the column `sku` because, is a very sparse and specific columns, BUT, the challenge request a forecast by product, so, i decide to include this column, but, for real problem proyect, as i mention before, i need more date range for decrease the BIAS. for this case, i one hot encode this column.

#### Insights:
1. The Column `macroCategory` got 13 different categories, but, just 6 represent the pareto distribution (80%) -> `['Bebidas no alcohólicas','Dulces y pasabocas (Botanas)','Cuidado del hogar','Licores y cigarrillos','Cuidado personal','Canasta básica']`
2. The column with more correlation with totalSales (my dependent varibale) is `stock` (positive correlation -> 36%), the column `unit_price_USD` have a negative correlation BUT, very weak. 
3. As the plot shows, the correlation of the independent variables with totalSales aint linear, so, 1st i start with a linearRegression baseline, but, then, i try a tree based model (RandomForestRegressor)
4. The Features,as expected, that have more power of explainability was `unit_price_USD` and `stock` according with the feature_importance generated for the RandomForestRegressor.


#### How to use the model?
1. Read the pickle model (serializable python object), and adjust the periods and the sku of the products. this will generate de predictions for the time frame that we want.
2. Personally i would suggest an API, that recieve the data as a json object, and in the backend, load the pickle, generate the prediction and return a json with the prediction.

#### Personal recomendations for improve the model:
1. if we have the last 2 years of sales, with all the products available we can run a stronger model, decreasing the `MSE(Mean Squared Error)` and `RMSE(Root Mean Squared Error)` and increasing the `R Square` and `Adjusted R Square`.
2. Add new Features, like: `holidays`, `weather`, `unemployment rate` and `inflation` maybe can reduce the bias and improve the metrics mentioned.
