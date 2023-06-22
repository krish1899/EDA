import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    st.set_page_config(layout="wide")
    EDA_tasks = ["1.distinguish attributes", "2.data cleaning", "3.plots","4.relationship analysis"]
    choice = st.sidebar.selectbox("select tasks", EDA_tasks)
    file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
    data = st.file_uploader("UPLOAD A DATASET 	:open_file_folder: ")
    if data:
        if file_format == 'csv':
            df = pd.read_csv(data)
        else:
            df = pd.read_excel(data)
        st.dataframe(df.head())

    if choice == '1.distinguish attributes':
        st.subheader(" distinguishing attributes in EDA :1234:")
        if st.checkbox("Show Shape"):
            if data is not None:
                st.write("rows and columns formate ", df.shape)

        if st.checkbox("Show Columns"):
            all_columns = df.columns.to_list()
            st.write(all_columns)

        if st.checkbox("Summary"):
            st.write(df.describe())

        if st.checkbox("Show Selected Columns"):
            all_columns = df.columns.to_list()
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("show numerical variables"):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            newdf = df.select_dtypes(include=numerics)
            st.dataframe(newdf)

        if st.checkbox("show categorical variables"):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            newdf = df.select_dtypes(include=numerics)
            df1 = df.drop(newdf, axis=1)
            st.dataframe(df1)

        if st.checkbox("percentage distribution of unique values in fields"):
            all_columns = df.columns.to_list()
            sel_cols = st.multiselect("Select Columns", all_columns)
            cd = df[sel_cols].value_counts(normalize = True)*100
            st.dataframe(cd)

    elif choice == '2.data cleaning':
        st.subheader(" data cleaning in EDA :hammer_and_wrench:")
        if st.checkbox("show the na values"):
            nas = df.isnull().sum()
            st.dataframe(nas)
        if st.checkbox("fill the missing values"):
            if st.checkbox("fill with median"):
                all_columns = df.columns.to_list()
                sens = st.multiselect("Select Columns", all_columns)
                selos = df[sens]
                dfm = df.fillna(selos.median())
                st.dataframe(dfm)
                st.download_button(label = 'download CSV',data = dfm.to_csv(),mime = 'text/csv')

            if st.checkbox("fill with mean"):
                all_columns = df.columns.to_list()
                sen = st.multiselect("Select Columns", all_columns)
                selo = df[sen]
                dfmea = df.fillna(abs(selo.mean()))
                st.dataframe(dfmea)
                st.download_button(label = 'download CSV',data = dfmea.to_csv(),mime = 'text/csv')

            if st.checkbox("fill with zeroes"):
                all_columns = df.columns.to_list()
                sen = st.multiselect("Select Columns", all_columns)
                se = df[sen]
                dfz = df.fillna(0)
                st.dataframe(dfz)
                st.download_button(label='download CSV', data=dfz.to_csv(), mime='text/csv')

        if st.checkbox("Remove duplicate values"):
            all_columns = df.columns.to_list()
            sel_mns = st.multiselect("Select Columns", all_columns)
            bd = df[sel_mns]
            dfb = bd.drop_duplicates()
            st.dataframe(dfb)

        if st.checkbox("detect outliers"):
            if st.checkbox("by z-score"):
                all_columns = df.columns.to_list()
                sel_mns = st.multiselect("Select Columns", all_columns)
                sel_nms = df[sel_mns]
                mean = np.mean(sel_nms)
                sd = np.std(sel_nms)
                upper = (mean + 3 *sd)
                lower = (mean - 3 *sd)
                upper_arr = np.extract(sel_nms > upper, sel_nms)
                lower_arr = np.extract(sel_nms < lower, sel_nms)
                tota = np.concatenate((upper_arr, lower_arr))
                st.dataframe(tota)
                upp_arr = np.where(df[sel_mns] > upper)[0]
                low_arr = np.where(df[sel_mns] < lower)[0]
                if st.button("remove outliers"):
                    ddm = df.copy(deep = True)
                    ddm.drop(index=upp_arr,inplace=True)
                    ddm.drop(index=low_arr, inplace=True)
                    st.dataframe(ddm)
            if st.checkbox("by iqr score"):
                all_columns = df.columns.to_list()
                sel_ns = st.multiselect("Select Columns", all_columns)
                sel_nos = df[sel_ns]
                q1 = np.percentile(sel_nos, 25)
                q3 = np.percentile(sel_nos, 75)
                iqr = abs(q3)-abs(q1)
                st.write(iqr)
                upper = abs(q3)+1.5*iqr
                lower = abs(q1)-1.5*iqr
                st.write("upper limit (or) 75 percentile  = ", upper)
                st.write("lower limit (or) 25 percentile  = " ,lower)
                upper_arr = np.extract(sel_nos >= upper,sel_nos)
                lower_arr = np.extract(sel_nos <= lower,sel_nos)
                tot = np.concatenate((upper_arr,lower_arr))
                st.dataframe(tot)
                upp_arr = np.where(df[sel_ns] >= upper)[0]
                low_arr = np.where(df[sel_ns] <= lower)[0]
                if st.button("remove outliers"):
                    ddf = df.copy(deep = True)
                    ddf.drop(index=upp_arr,inplace=True)
                    ddf.drop(index=low_arr, inplace=True)
                    st.dataframe(ddf)
        if st.checkbox("normalisation"):
            all_columns = df.columns.to_list()
            sel_mns = st.multiselect("Select Columns", all_columns)
            sems = df[sel_mns]
            result = sems.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
            st.dataframe(result)
            st.download_button(label='download CSV', data=result.to_csv(), mime='text/csv')

        if st.checkbox("standardisation"):
            all_columns = df.columns.to_list()
            sel_mns = st.multiselect("Select Columns", all_columns)
            semsi = df[sel_mns]
            resulti = semsi.apply(lambda d: ((d-np.mean(d))/np.std(d)))
            st.dataframe(resulti)
            st.download_button(label='download CSV', data=resulti.to_csv(), mime='text/csv')
    elif choice == '3.plots':
        st.subheader("plots    :bar_chart:")
        if data is not None:
            data.seek(0)
            if st.checkbox("area chart", key=1):
                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                df[selected_cs].value_counts().plot(kind="area", color=user_colour)
                st.pyplot(fig)
            if st.checkbox("bar chart", key=2):
                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                df[selected_cs].value_counts().plot(kind="bar", color=user_colour)
                st.pyplot(fig)
            if st.checkbox("hist", key=3):
                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig1 = plt.figure()
                df[selected_cs].value_counts().plot(kind="hist", color=user_colour)
                st.pyplot(fig1)
            if st.checkbox("lineplot", key=4):
                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                df[selected_cs].value_counts().plot(kind="line", color=user_colour)
                st.pyplot(fig)
            if st.checkbox("kde plot", key=5):
                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                df[selected_cs].value_counts().plot(kind="kde", color=user_colour)
                st.pyplot(fig)


    elif choice == "4.relationship analysis" :
        st.subheader("analyzing the relationship between field")
        if data is not None:
            data.seek(0)
        if st.checkbox('correlation analysis'):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            nef = df.select_dtypes(include=numerics)
            fig, ax = plt.subplots()
            sns.heatmap(nef.corr(), ax=ax)
            st.write(fig)
        if st.checkbox('relation plot'):
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            nef = df.select_dtypes(include=numerics)
            fig=sns.pairplot(nef.corr())
            st.pyplot(fig)

if __name__ == '__main__':
    main()
