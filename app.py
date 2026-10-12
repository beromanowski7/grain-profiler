import io
import pandas as pd
import plotly.express as px
import streamlit as st
from profiler import analyze

st.set_page_config(page_title='GRAIN Profiler', page_icon='🔎', layout='wide')
st.title('🔎 GRAIN Profiler')
st.caption('Exploratory dataset profiling • observed relationships are not causal or universal business rules')
file = st.file_uploader('Upload an Excel workbook or CSV', type=['xlsx', 'csv'])
if file is None:
    st.info('Upload a dataset to begin. Files are processed in this running application; no upload is sent to GitHub.')
    st.stop()
try:
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        data = file.getvalue()
        sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
        sheet = st.selectbox('Worksheet', sheets)
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet)
except Exception as exc:
    st.error(f'Unable to read file: {exc}')
    st.stop()

with st.sidebar:
    st.header('Analysis settings')
    max_cols = st.slider('Maximum relationship columns', 2, 40, 20)
    max_key_size = st.slider('Composite key size', 1, 3, 2)
    time_col = st.selectbox('Chronological validation (optional)', ['None'] + [str(c) for c in df.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns])
    st.warning('Never publish confidential, ITAR-controlled, or proprietary datasets to a public repository.')

st.write(f'**{len(df):,} rows × {len(df.columns)} columns**')
with st.expander('Preview source rows'):
    st.dataframe(df.head(30), use_container_width=True)
with st.spinner('Computing diagnostics...'):
    try:
        results = analyze(df, max_cols=max_cols, max_key_size=max_key_size,
                          chronological=None if time_col == 'None' else time_col)
    except Exception as exc:
        st.error(f'Analysis failed: {exc}')
        st.stop()

tabs = st.tabs(['Column traits', 'Dependency heatmap', 'Functional dependencies', 'Candidate keys', 'Held-out validation', 'Export'])
with tabs[0]:
    st.dataframe(results['profile'], use_container_width=True)
with tabs[1]:
    st.write('**Row = predictor X; column = target Y.** Values measure the fraction of target entropy explained.')
    mat = results['matrix']
    fig = px.imshow(mat, zmin=0, zmax=1, aspect='auto', color_continuous_scale='Teal',
                    labels={'x':'Target Y', 'y':'Predictor X', 'color':"Theil's U"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Diagonal omitted. Constant targets are undefined. High-cardinality identifiers can score artificially high.')
with tabs[2]:
    st.dataframe(results['dependencies'], use_container_width=True)
    st.caption('Exact = zero conflicting predictor values in THIS dataset. Row agreement is an in-sample mapping measure.')
with tabs[3]:
    st.dataframe(results['keys'], use_container_width=True)
    st.caption(f"Checked {results['key_combinations_checked']:,} combinations. Truncated: {results['key_search_truncated']}. Candidate keys require business validation.")
with tabs[4]:
    st.dataframe(results['validation'], use_container_width=True)
    st.caption('Simple train/test majority lookup. Accuracy is not statistical confidence. Compare against baseline and key coverage; repeated entities can cause leakage.')
with tabs[5]:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        for name, key in [('Column Profile','profile'), ('Dependency Matrix','matrix'),
                          ('Dependency Audit','dependencies'), ('Candidate Keys','keys'),
                          ('Heldout Validation','validation')]:
            results[key].to_excel(writer, sheet_name=name, index=(key == 'matrix'))
        pd.DataFrame({'note': ['Observed patterns are not proven business rules.',
                               'Theil U is entropy reduction, not prediction confidence.',
                               'Validation accuracy depends on split design and key coverage.',
                               'Numeric measurements are excluded from the categorical dependency matrix by default.',
                               'Key search and validation are capped for interactive responsiveness.']}).to_excel(writer, sheet_name='Interpretation', index=False)
    st.download_button('Download evidence workbook', data=out.getvalue(),
                       file_name='grain_evidence.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
