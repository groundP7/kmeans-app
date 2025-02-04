import streamlit as st
import pandas as pd

def main() :
    st.title("K-Means Clustering App")

    # 1. csv file upload
    file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    
    if file is not None :
        # 2. 데이터 불러오기
        df = pd.read_csv(file)
        st.dataframe(df.head())

        # 3. 유저가 컬럼을 선택할수 있게 한다.
        st.info("K-Means 클러스터링에 사용할 컬럼을 선택해주세요.")
        selected_columns = st.multiselect("컬럼 선택", df.columns)

if __name__ == "__main__" :
    main()