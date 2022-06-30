import streamlit as st
import pandas as pd
from Doc2Vec import similar_doc2vec

df=pd.read_excel('삼성전자_result.xlsx')

st.title("""Doc2Vec 구현""")
st.write("""Doc2Vec""")

url = st.text_input('기사 주소를 입력해주세요', '기사주소')
st.write('입력한 기사의 주소는', url)

if st.button('기사 검색'):
    st.write('입력한 기사의 비슷한 주소는', similar_doc2vec(df,url))


st.dataframe(df)