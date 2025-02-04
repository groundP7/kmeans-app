import streamlit as st
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
import platform

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Linux':
    plt.rcParams['font.family'] = 'NanumGothic'


@st.cache_data
def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)


plt.rcParams['axes.unicode_minus'] = False

def main() :

    fontRegistered()
    plt.rc('font', family='NanumGothic')

    st.title("K-Means Clustering App")

    # 1. csv file upload
    file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    
    if file is not None :
        # 2. 데이터 불러오기
        df = pd.read_csv(file)
        st.dataframe(df.head())
        # 비어있는 데이터가 있는지 확인하기
        st.info("Nan 이 있으면 행을 삭제합니다.")
        st.dataframe(df.isna().sum())
        # 비어있는 데이터 삭제하기
        df.dropna(inplace= True)
        df.reset_index(drop= True, inplace= True)

        # 3. 유저가 컬럼을 선택할수 있게 한다.
        st.info("K-Means 클러스터링에 사용할 컬럼을 선택해주세요.")
        selected_columns = st.multiselect("컬럼 선택", df.columns)

        if len(selected_columns) == 0 :
            return

        df_new = pd.DataFrame()
        # 4. 문자열인지 숫자인지 확인.
        for column in selected_columns :
            if is_integer_dtype(df[column]) :
                print(column + ' : int')
                print(df[column])
                df_new[column] = df[column]
                print(df_new[column])
            elif is_float_dtype(df[column]) :
                print(column + ' : float')
                df_new[column] = df[column]
            elif is_object_dtype(df[column]) :
                print(column + ' : object')
                if df[column].nunique() <= 2 :
                    # 레이블 인코딩
                    encoder = LabelEncoder()
                    df_new[column] = encoder.fit_transform(df[column])
                else :
                    # 원핫 인코딩
                    ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])] , remainder="passthrough")
                    # 컬럼이름을 정렬후 유니크처리
                    column_names = sorted(df[column].unique())
                    df_new[column_names] = ct.fit_transform(df[column].to_frame())
            else :
                st.text(f"{column} 컬럼은 K-Means에 사용 불가하므로 제외하겠습니다.")
        
        st.info("K-Means를 수행하기 위한 데이터 프레임 입니다.")
        st.dataframe(df_new)

        st.subheader("최적의 k값을 찾기위해 WCSS를 구합니다.")
        
        # 데이터의 갯수가 클러스터링 갯수보다는 크거나 같아야 하므로
        # 해당 데이터의 갯수로 최대 k값을 정한다
        st.text(f"데이터의 갯수는 {df_new.shape[0]}개 입니다.")
        if df_new.shape[0] < 10 :
            max_k = st.slider("k값 선택(최대 그룹갯수)", min_value= 2, max_value= df_new.shape[0])
        else :
            max_k = st.slider("k값 선택(최대 그룹갯수)", min_value= 2, max_value= 10)

        wcss = []
        for k in range(1, max_k+1) :
            kmeans = KMeans(n_clusters= k, random_state= 4)
            kmeans.fit(df_new)
            wcss.append(kmeans.inertia_)

        fig1 = plt.figure()
        plt.plot(range(1, max_k+1), wcss)
        plt.title("The Elbow Method")
        plt.xlabel("클러스터의 수")
        plt.ylabel("WCSS 값")
        st.pyplot(fig1)

        st.text("원하는 클러스터링(그룹) 갯수를 입력하세요")
        k = st.number_input("숫자 입력", min_value=2, max_value=max_k)

        kmeans = KMeans(n_clusters= k, random_state= 4)
        df["Group"] = kmeans.fit_predict(df_new)

        st.info("그룹 정보가 저장 되었습니다.")
        st.dataframe(df)

if __name__ == "__main__" :
    main()