import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
# NOTE: All necessary ML imports are placed here
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import make_pipeline 

# --- UI ENHANCEMENTS: CUSTOM CSS AND COLOR PALETTE (PROFESSIONAL LIGHT THEME) ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* Light Theme Colors (Clean, Professional, Tech/Industrial Look) */
        :root {
            --primary-bg: #FFFFFF;    /* Primary Background (White) */
            --secondary-bg: #F0F2F6;  /* Secondary Background (Light Ice Gray) */
            --accent-color: #0E77B5;  /* Deep Blue Accent */
            --border-color: #E0E0E0;  /* Clean Light Gray Border */
            --text-color: #333333;    /* Dark Text */
            --success-color: #388E3C; /* Medium Green */
            --warning-color: #FF9900; /* Amber Warning/Highlight */
        }
        
        /* Apply Primary Background to the main app container */
        .stApp {
            background-color: var(--primary-bg);
            color: var(--text-color);
        }
        
        /* Apply Secondary Background to Sidebar, Tabs, and Input Containers */
        .st-emotion-cache-1cypcdb, .st-emotion-cache-p5m9b9, .st-emotion-cache-17z5936, .st-emotion-cache-1r6dm1 {
             background-color: var(--secondary-bg); 
        }

        /* Set default text color */
        p, label, .stMarkdown {
            color: var(--text-color) !important;
        }

        /* Headers and Titles (Deep Blue Accent) */
        h1, h2, h3, h4, h5, h6 {
            color: var(--accent-color) !important;
        }
        
        /* Primary Buttons/Accent */
        .stButton>button {
            background-color: var(--accent-color);
            color: white !important; /* Force button text to white */
            border-radius: 8px;
            border: 1px solid var(--accent-color);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #005A8C; /* Slightly darker blue on hover */
            border-color: #005A8C;
        }

        /* Info/Warning Boxes (Clean look on light mode) */
        .stAlert {
            border-left: 5px solid var(--accent-color); 
            background-color: var(--secondary-bg); /* Light card background */
            border-radius: 8px;
            color: var(--text-color);
        }
        
        /* Metric Box Customization (Clean light cards) */
        .st-emotion-cache-10o1d10 {
            background-color: var(--secondary-bg); /* Light background for metrics */
            border-radius: 10px;
            padding: 10px 15px;
            border: 1px solid var(--border-color); /* Subtle border */
        }

        /* Dataframes */
        .st-emotion-cache-135ziuz {
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }

        /* Header Divider */
        h2 {
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 5px;
            margin-bottom: 15px;
            color: var(--text-color);
        }
        
        </style>
    """, unsafe_allow_html=True)

# Apply CSS immediately
apply_custom_css()


# --- PAGE CONFIG ---
st.set_page_config(page_title="Cummins Engine Data Dashboard", page_icon="🚛", layout="wide")
st.title("🚛 Cummins Engine Data Analysis Dashboard")

# --- Helper: Dynamic Header Detection Function ---
def find_header_row(file_path):
    """Finds the row index where the actual column headers begin."""
    # Attempt common encodings
    for encoding in ['ISO-8859-1', 'utf-8', 'latin-1']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if line.strip().startswith('"Date","Time"') or line.strip().startswith('Date,Time') or ('ID' in line and 'Frame' in line):
                        return i
            # If we successfully read the file but didn't find the header, return 0 (start)
            return 0 
        except UnicodeDecodeError:
            # Continue to the next encoding if this one fails
            continue
        except Exception:
            # Handle other file-related errors
            return -1 
    return -1 # Should only happen if all encodings fail

# --- Helper: Robust Data Loader (FINALIZED WITH ALL FEATURES) ---
@st.cache_data
def load_data(uploaded_file):
    df = None
    error_message = None
    
    try:
        file_path = f"temp_{uploaded_file.name}"
        # Save the uploaded file temporarily
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        header_row = find_header_row(file_path)

        if header_row == -1:
            return None, "File reading failed: Could not determine file header or encoding."

        # Load data dynamically
        try:
            # Try to read with ISO-8859-1 encoding, which is common for CSV logs
            skip_count = header_row 
            df = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1', skiprows=skip_count)
            
            # Additional check if header detection failed due to format issues
            if 'Date' not in df.columns and 'ID' not in df.columns:
                df = pd.read_csv(file_path, low_memory=False, encoding='ISO-8859-1', skiprows=skip_count, header=0)

            # Drop first unnamed column if present
            if df.columns[0].startswith('Unnamed:'):
                df = df.iloc[:, 1:] 
            
        except Exception as e:
            return None, f"PD Read Error: {e}"

        # --- Data Processing and ML Prep ---
        if df is not None:
            # 1. Detect File Type: CAN Log
            if 'ID' in df.columns and 'Frame' in df.columns:
                df['ID'] = df['ID'].astype(str).str.replace('"', '').str.strip()
                if 'system time' in df.columns:
                    df['system time'] = df['system time'].astype(str).str.replace('="', '', regex=False).str.replace('"', '', regex=False)
                    df.rename(columns={'system time': 'Timestamp'}, inplace=True) 
                return df, "Raw CAN Log"
                
            # 2. Detect File Type: INSITE Sensor Data 
            elif 'Engine Speed (RPM)' in df.columns:
                
                # Create a map to store original column names for look-up during encoding
                original_col_map = {} 
                
                # Combine Date and Time into a single datetime column
                if 'Date' in df.columns and 'Time' in df.columns:
                    try:
                        df['DateTimeStr'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
                        df['Timestamp'] = pd.to_datetime(df['DateTimeStr'], errors='coerce')
                        df.drop(columns=['DateTimeStr'], inplace=True)
                    except:
                        st.warning("Could not parse Date/Time columns; using Log Index.")
                
                # FINAL DATA PROCESSING LOOP: Selective Coercion and Label Encoding
                for col in df.columns:
                    if col not in ['Timestamp', 'Date', 'Time', 'Log Index']:
                        
                        series = pd.to_numeric(df[col], errors='coerce')
                        numeric_percentage = series.notna().sum() / len(df)
                        
                        if numeric_percentage > 0.5:
                            df[col] = series
                        else:
                            if df[col].dtype == 'object':
                                # Store original status column *before* encoding it
                                original_col_map[col] = df[col] 

                                encoded_col_name = f"{col} (Encoded)"
                                df[encoded_col_name] = df[col].astype('category').cat.codes
                                
                # Store the map of original text columns in session state/cache for lookup
                st.session_state['original_category_map'] = original_col_map

                return df, "INSITE Sensor Data"
                
            else:
                return df, "Unknown"

    except Exception as e:
        # Catch any unexpected error during file reading or processing
        return None, f"General Data Processing Error: {e}"
    
    return None, "File processing failed."


# --- Helper: Normalization ---
def normalize_column(df, col_name):
    """Scales a single column between 0 and 1."""
    max_val = df[col_name].max()
    min_val = df[col_name].min()
    if max_val != min_val:
        return (df[col_name] - min_val) / (max_val - min_val)
    return df[col_name]

# --- SIDEBAR: 1. UPLOAD ---
st.sidebar.header("1. Data Source")
uploaded_files = st.sidebar.file_uploader("Upload CSV File(s)", type=["csv"], accept_multiple_files=True)

# --- MAIN APP ---
if 'original_category_map' not in st.session_state:
    st.session_state['original_category_map'] = {}

if uploaded_files:
    primary_file = uploaded_files[0]
    status_container = st.container()

    with st.spinner(f"Processing {primary_file.name}..."):
        df_result, file_type_or_error = load_data(primary_file)

    if df_result is not None and len(df_result) > 0:
        df_global = df_result.copy() 
        file_type = file_type_or_error
        
        status_container.success(f"✅ Baseline File: **{primary_file.name}** ({file_type}) loaded.")

        if file_type == "Raw CAN Log" and len(uploaded_files) > 1:
            st.title("📡 Multi-File CAN Differential Analysis")
            can_tab1, can_tab2 = st.tabs(["📊 Traffic Overview", "🔍 ID Fingerprinting"])
            
            with can_tab1:
                st.subheader(f"Baseline Traffic Analysis ({primary_file.name})")
                id_counts = df_global['ID'].value_counts().reset_index()
                id_counts.columns = ['CAN ID', 'Count']
                fig = px.bar(id_counts.head(20), x='CAN ID', y='Count', title="Top 20 Active IDs", color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

            with can_tab2:
                st.header("🔍 Sensor Identification (Fingerprinting)")
                st.info(f"Baseline: {primary_file.name}")
                baseline_ids = set(df_global['ID'].unique())
                
                for i in range(1, len(uploaded_files)):
                    target_file = uploaded_files[i]
                    target_df, t_type = load_data(target_file)
                    st.write(f"---")
                    st.subheader(f"📂 Comparison with: {target_file.name}")
                    
                    if t_type == "Raw CAN Log":
                        target_ids = set(target_df['ID'].unique())
                        new_ids = target_ids - baseline_ids
                        
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.success(f"✨ Found {len(new_ids)} Unique Candidate IDs")
                            st.dataframe(pd.DataFrame({"Candidate CAN IDs": list(new_ids)}), use_container_width=True)
                        with col_res2:
                            st.warning("📈 Frequency Shift Analysis")
                            b_counts = df_global['ID'].value_counts(normalize=True)
                            t_counts = target_df['ID'].value_counts(normalize=True)
                            freq_delta = (t_counts - b_counts).sort_values(ascending=False).head(10).reset_index()
                            freq_delta.columns = ['CAN ID', 'Frequency Increase']
                            st.dataframe(freq_delta, use_container_width=True)
        
        else:
            st.sidebar.divider()
            st.sidebar.header("2. Global Filters")
            total_rows = len(df_global)
            df_filtered = df_global 
            
            numeric_cols = df_global.select_dtypes(include='number').columns.tolist()
            time_cols = ['Log Index']
            if 'Timestamp' in df_global.columns: time_cols.append('Timestamp')
            if 'Date' in df_global.columns: time_cols.append('Date')
            if 'Time' in df_global.columns: time_cols.append('Time')
            status_cols = [col for col in df_global.columns if df_global[col].dtype == 'object' and col not in time_cols]
            x_options = time_cols + status_cols + numeric_cols
            default_x_option = 'Timestamp' if 'Timestamp' in x_options else 'Log Index'
            
            if total_rows > 1:
                start_row, end_row = st.sidebar.slider("Select Data Range:", 0, total_rows, (0, total_rows))
                df_filtered = df_global.iloc[start_row:end_row].copy()
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Custom Plotter & Regression", "🧪 ML Analysis", "📄 Data & Export", "⚙️ System Info"])

            with tab1: # Custom Plotter Tab
                if file_type == "INSITE Sensor Data":
                    st.header("📈 Custom Plotting & Regression Analysis")
                    y_numeric_cols = [c for c in df_filtered.select_dtypes(include='number').columns.tolist() if c != 'Log Index']
                    
                    st.sidebar.divider()
                    st.sidebar.header("3. Plot Settings")
                    y_axis_val_list = st.sidebar.multiselect("Y-Axis:", y_numeric_cols, default=[y_numeric_cols[0]] if y_numeric_cols else None, key='plot_y_multi')
                    x_axis_val = st.sidebar.selectbox("X-Axis:", options=x_options, index=x_options.index(default_x_option) if default_x_option in x_options else 0)
                    chart_type = st.sidebar.radio("Chart Type:", ["Line Chart", "Scatter Plot"])
                    norm_x = st.sidebar.checkbox("Normalize X-Axis (0-1)", value=False)
                    norm_y = st.sidebar.checkbox("Normalize Y-Axis (0-1)", value=False)
                    reg_enabled = st.sidebar.checkbox("Show Regression Plot Below", value=False)

                    if y_axis_val_list:
                        plot_df = df_filtered.copy()
                        hover_data_list = []
                        
                        # --- FIX: HOVER LOGIC FOR CATEGORICAL LABELS AND NORMALIZED DATA ---
                        for col in y_axis_val_list:
                            # If it's an encoded categorical column
                            if " (Encoded)" in col:
                                original_name = col.replace(" (Encoded)", "")
                                # Pull the actual text label from original data for hover
                                plot_df[f"Label_{original_name}"] = df_global[original_name].iloc[plot_df.index]
                                hover_data_list.append(f"Label_{original_name}")
                            
                            # If normalization is active, keep original numeric for hover
                            if norm_y:
                                plot_df[f"Original_{col}"] = plot_df[col]
                                plot_df[col] = normalize_column(plot_df, col)
                                hover_data_list.append(f"Original_{col}")

                        if chart_type == "Line Chart": fig = px.line(plot_df, x=x_axis_val, y=y_axis_val_list, hover_data=hover_data_list)
                        else: fig = px.scatter(plot_df, x=x_axis_val, y=y_axis_val_list, hover_data=hover_data_list)
                        st.plotly_chart(fig, use_container_width=True)

                        if reg_enabled:
                            st.divider(); st.subheader("🧪 Regression Model Training")
                            col_y, col_x, col_model = st.columns(3)
                            with col_y: reg_target_col = st.selectbox("Target (Y):", y_numeric_cols, key='reg_target_tab1')
                            with col_x:
                                reg_feature_options = [c for c in y_numeric_cols if c != reg_target_col]
                                reg_feature_cols = st.multiselect("Features (X):", reg_feature_options, key='reg_feature_tab1')
                            
                            with col_model:
                                if len(reg_feature_cols) == 1: m_opts = ["Simple Linear", "Polynomial"]
                                elif len(reg_feature_cols) > 1: m_opts = ["Multiple Linear", "Polynomial"]
                                else: m_opts = ["Linear", "Polynomial"]
                                model_selection = st.selectbox("Model Type:", options=m_opts, key='reg_model_type')
                                poly_degree = st.number_input("Degree:", 2, 5, 2) if model_selection == "Polynomial" else 1

                            if st.button("Run Regression Model", key='run_reg_tab1_button', use_container_width=True):
                                if not reg_feature_cols: st.warning("Select Feature (X).")
                                else:
                                    df_reg = plot_df[[reg_target_col] + reg_feature_cols].dropna()
                                    X_train, X_test, Y_train, Y_test = train_test_split(df_reg[reg_feature_cols], df_reg[reg_target_col], test_size=0.3, random_state=42)
                                    
                                    if "Polynomial" in model_selection: model = make_pipeline(PolynomialFeatures(poly_degree, include_bias=False), LinearRegression())
                                    else: model = LinearRegression()
                                    
                                    model.fit(X_train, Y_train); Y_pred = model.predict(X_test)
                                    st.metric("R-squared Score", f"{model.score(X_test, Y_test):.4f}")
                                    
                                    lr = model.named_steps['linearregression'] if "Polynomial" in model_selection else model
                                    feats = model.named_steps['polynomialfeatures'].get_feature_names_out(reg_feature_cols) if "Polynomial" in model_selection else reg_feature_cols
                                    eq = f"{reg_target_col.replace(' ','')} = {lr.intercept_:.4f}"
                                    for c, f in zip(lr.coef_, feats): eq += f" {'+' if c > 0 else '-'} {abs(c):.4f} * {f.replace(' ', '*')}"
                                    st.code(eq)

                                    st.write("#### Actual vs. Predicted Value Plot")
                                    plot_data = pd.DataFrame({"Actual Value": Y_test, "Predicted Value": Y_pred})
                                    min_v, max_v = plot_data[['Actual Value', 'Predicted Value']].min().min(), plot_data[['Actual Value', 'Predicted Value']].max().max()
                                    fig_reg = px.scatter(plot_data, x="Actual Value", y="Predicted Value", title="Regression Performance")
                                    fig_reg.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="Red", width=2, dash="dash"))
                                    st.plotly_chart(fig_reg, use_container_width=True)

                elif file_type == "Raw CAN Log":
                    st.header("📡 CAN Message Frequency Analysis")
                    id_counts = df_filtered['ID'].value_counts().reset_index()
                    id_counts.columns = ['CAN ID', 'Count']
                    fig = px.bar(id_counts.tail(20), x='Count', y='CAN ID', orientation='h', title="Top 20 IDs")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2: # ML Analysis
                if file_type == "INSITE Sensor Data":
                    st.header("🧪 Feature Correlation Analysis")
                    ml_num = [c for c in df_global.select_dtypes(include='number').columns if c != 'Log Index']
                    feat_corr = st.multiselect("Select Features:", ml_num, default=ml_num[:5])
                    if feat_corr:
                        fig_corr = px.imshow(df_global[feat_corr].corr(), text_auto=".2f", color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig_corr, use_container_width=True)

            with tab3: # Data Export
                st.subheader("📄 Filtered Data Inspector")
                st.dataframe(df_filtered, use_container_width=True)
                st.download_button("📥 Download Filtered CSV", df_filtered.to_csv(index=False).encode('utf-8'), "filtered_data.csv")

            with tab4: # Info
                st.subheader("⚙️ File Metadata")
                st.metric("Detected File Type", file_type)
                st.metric("Total Rows", f"{total_rows:,}")

    elif uploaded_files and file_type_or_error:
        st.error(f"Data Loading Failed: {file_type_or_error}")
else:
    st.info("👋 Please upload a file to begin.")