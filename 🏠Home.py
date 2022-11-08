import streamlit as st 



st.set_page_config(page_title='Chiper Test',layout='wide',page_icon="./img/ch.png")


col1_,col2_,col3_ = st.columns(3)
with col1_:
    st.image('./img/chiper_logo.png')
st.write('---') 
col1,col2,col3 = st.columns(3)
with col1:
    st.markdown('## Chiper: Data Scientist Test')

with col3:
    st.markdown('#### Javier Javier Daza Olivella - Sr. Data Scientist')
    st.write('https://www.linkedin.com/in/javier-daza')


st.write('---') 
st.markdown('### The Challenge 📝')
with st.expander('Expand here >>', expanded = False):
    st.markdown('''
**Chiper** es una Startup tecnológica presente en Colombia, México y Brasil. A través
de su plataforma de comercio electrónico, ofrece productos de consumo masivo a
precios directos de fabricante a comercios independientes como tiendas de barrio,
licorerías, minimercados, entre otros.
Al ser una startup B2C debe tener procesos optimos de abastecimiento, usted en
esta prueba debe diseñar una lógica de abastecimiento para el caso de uso de
una bodega.
En el archivo.xlsx se contiene información del mes de enero-febrero del 2022 para
nuestra bodega de la ciudad de Medellín, este archivo contiene los siguientes
campos:
- Sku
- Nombre de sku
- Numero de unidades vendidas
- Fecha
- WarehouseId
- Macrocategoria
- Categoría
- Numero de stock  en bodega al finalizar el dia
- Usdtotal
- Unidades vendidas con actividad comercial
- locationId

*Nota: actividades comerciales son descuentos que se tienen en ciertos productos.*

#### ¿Qué se requiere?

Se requiere el recomendado de ordenes de compra que se deben realizar a
nuestros proveedores,  para cada uno de nuestros productos en los primeros 15
dias del mes de marzo del 2022.
##### Esto debe tener las siguientes asunciones:
1. El leadtime de los proveedores para productos tipo A es de 2 días.
2. El leadtime de los proveedores para productos tipo B es de 5 días
3. El leadtime de los proveedores para productos tipo C es de 10 días.


Asuma que los productos tipo A se pueden ordenar todos los días, tipo B y C se
pueden  ordenar Martes, Jueves y Viernes.

*Nota: el leadtime es el tiempo en días en cual un proveedor me puede entregar. La
clasificación de productos A,B, C se puede basar en el 80% de la venta, 10% y
10% correspondientes.*

**Se debe realizar una lógica de abastecimiento esta debe incluir, forecast (mediciones), lógicas de recomendado y cualquier análisis exploratorio que crea pertinente sobre nuestra data. Al finalizar debe realizar una presentación con los resultados obtenidos.**

*Nota: usted puede asumir ciertas condiciones dentro del desarrollo de la prueba*
    
    
    
    
    ''')


            