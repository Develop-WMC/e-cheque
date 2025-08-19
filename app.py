import streamlit as st
import os
import pandas as pd
import tempfile
import base64
import json
import toml
from datetime import datetime, timedelta
import time
from io import BytesIO
import zipfile
import sqlite3

# Import components directly from files
import gmail_component
import processing_component
import teams_component

# Set page config with a more appealing icon
st.set_page_config(
    page_title="e-Cheque Processing Pipeline",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom styling - YOUR ORIGINAL CSS IS PRESERVED
st.markdown("""
<style>
    /* Core styling */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .step-header {
        font-size: 1.4rem;
        font-weight: bold;
        margin: 0.8rem 0;
    }
    
    .subheader {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.6rem 0;
    }
    
    /* Status boxes */
    .success-box {
        padding: 0.8rem;
        background-color: rgba(52, 168, 83, 0.2);
        border-left: 4px solid #34A853;
        margin-bottom: 0.8rem;
    }
    
    .warning-box {
        padding: 0.8rem;
        background-color: rgba(251, 188, 4, 0.2);
        border-left: 4px solid #FBBC04;
        margin-bottom: 0.8rem;
    }
    
    .info-box {
        padding: 0.8rem;
        background-color: rgba(66, 133, 244, 0.2);
        border-left: 4px solid #4285F4;
        margin-bottom: 0.8rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4285F4;
        color: white;
    }
    
    /* Input fields */
    .stTextInput > div > div > input, 
    .stDateInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: white;
        border: 1px solid #ccc;
    }

    /* Date input specific styling */
    .stDateInput {
        background-color: white;
    }

    .stDateInput > div {
        background-color: white;
    }

    .stDateInput input[type="date"] {
        background-color: white;
    }

    /* Make form fields and buttons stand out */
    button, .stButton button, .stDownloadButton button {
        background-color: #4285F4;
        color: white;
    }
    
    /* Status alerts */
    .stAlert > div {
        background-color: #f8f9fa;
    }
    
    .stInfo > div {
        background-color: rgba(66, 133, 244, 0.1);
        border-left-color: #4285F4;
    }
    
    .stSuccess > div {
        background-color: rgba(52, 168, 83, 0.1);
        border-left-color: #34A853;
    }
    
    .stWarning > div {
        background-color: rgba(251, 188, 4, 0.1);
        border-left-color: #FBBC04;
    }
    
    .stError > div {
        background-color: rgba(234, 67, 53, 0.1);
        border-left-color: #EA4335;
    }

    /* Calendar styling */
    .streamlit-expanderContent {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load config from secrets.toml
def load_config():
    if hasattr(st, 'secrets'):
        return st.secrets
    config_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return toml.load(f)
    return {"gmail": {}, "teams": {}, "gemini": {}}

# --- NEW: HELPER FUNCTIONS FOR MANAGING THE MAPPING CSV ---
MAPPING_FILE = "payee_mappings.csv"
MAPPING_COLUMNS = ['Payee', 'Teams_Folder', 'GL_Code']

def load_mapping_data():
    """Loads the payee mapping data from the CSV file."""
    if not os.path.exists(MAPPING_FILE):
        df = pd.DataFrame(columns=MAPPING_COLUMNS)
        df.to_csv(MAPPING_FILE, index=False)
        return df
    try:
        return pd.read_csv(MAPPING_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    except Exception as e:
        st.error(f"Critical error loading mapping file: {e}")
        return pd.DataFrame(columns=MAPPING_COLUMNS)

def save_mapping_data(df):
    """Saves the mapping DataFrame to the CSV file."""
    try:
        df.dropna(subset=['Payee'], inplace=True)
        df.drop_duplicates(subset=['Payee'], keep='last', inplace=True)
        df.to_csv(MAPPING_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to save mapping file: {e}")
        return False
# --- END OF NEW HELPER FUNCTIONS ---

# Database functions for persistent storage
def init_db():
    conn = sqlite3.connect('echeque_processing.db', isolation_level=None)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS processed_files (
        filename TEXT PRIMARY KEY,
        processed_date TEXT,
        data TEXT
    )
    ''')
    conn.close()

def load_from_db():
    conn = sqlite3.connect('echeque_processing.db')
    c = conn.cursor()
    c.execute("SELECT filename FROM processed_files")
    filenames = {row[0] for row in c.fetchall()}
    c.execute("SELECT data FROM processed_files")
    files_data = []
    for row in c.fetchall():
        file_data = json.loads(row[0])
        binary_fields = ['content', 'pdf', 'original_pdf', 'pdf_data']
        for field in binary_fields:
            field_base64_key = f'_{field}_is_base64'
            if field_base64_key in file_data and file_data[field_base64_key]:
                if field in file_data:
                    file_data[field] = base64.b64decode(file_data[field])
                del file_data[field_base64_key]
        files_data.append(file_data)
    conn.close()
    return filenames, files_data

def save_to_db(processed_file):
    conn = sqlite3.connect('echeque_processing.db')
    c = conn.cursor()
    filename = processed_file.get('original_filename') or processed_file.get('generated_filename')
    processed_date = datetime.now().isoformat()
    storage_file = processed_file.copy()
    binary_fields = ['content', 'pdf', 'original_pdf', 'pdf_data']
    for field in binary_fields:
        if field in storage_file and isinstance(storage_file[field], bytes):
            storage_file[field] = base64.b64encode(storage_file[field]).decode('utf-8')
            storage_file[f'_{field}_is_base64'] = True
    try:
        data_json = json.dumps(storage_file)
    except TypeError as e:
        problematic_keys = []
        for key, value in storage_file.items():
            try:
                json.dumps({key: value})
            except TypeError:
                problematic_keys.append(f"{key} (type: {type(value)})")
        raise TypeError(f"Cannot JSON serialize these keys: {', '.join(problematic_keys)}") from e
    c.execute(
        "INSERT OR REPLACE INTO processed_files (filename, processed_date, data) VALUES (?, ?, ?)",
        (filename, processed_date, data_json)
    )
    conn.commit()
    conn.close()

def create_zip_from_files(files):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for file in files:
            zip_file.writestr(file['filename'], file['content'])
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

config = load_config()
init_db()

if 'downloaded_files' not in st.session_state:
    st.session_state.downloaded_files = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'processed_filenames' not in st.session_state:
    filenames, files_data = load_from_db()
    st.session_state.processed_filenames = filenames
    st.session_state.processed_files = files_data

st.markdown('<div class="main-header">e-Cheque Processing Pipeline</div>', unsafe_allow_html=True)
st.markdown("""
This application streamlines the e-cheque processing workflow. Use the tabs below to navigate through the steps.
""")

# --- MODIFIED: Added a new tab for Mapping Management ---
tabs = st.tabs([
    "📩 Step 1: Download", 
    "🔍 Step 2: Process", 
    "📤 Step 3: Upload", 
    "📂 Mapping Management"
])

# STEP 1: DOWNLOAD TAB - FULL ORIGINAL CODE RESTORED
with tabs[0]:
    st.markdown('<div class="step-header">Step 1: Download e-Cheques from Gmail</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This step connects to your Gmail account to search for and download e-cheque attachments.
    Alternatively, you can upload e-cheque PDFs directly.
    """)
    
    st.markdown('<div class="subheader">Gmail API Configuration</div>', unsafe_allow_html=True)
    st.info("""
    Gmail API credentials are configured from secrets.toml
    - Client ID: ✓ Configured
    - Client Secret: ✓ Configured 
    - Token: ✓ Configured
    """)
    
    st.markdown('<div class="subheader">Email Search Criteria</div>', unsafe_allow_html=True)
    
    with st.form(key="email_form"):
        st.markdown("**Set date range to search for e-cheque emails:**")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now().date())
        with col2:
            end_date = st.date_input("End Date", (datetime.now() + timedelta(days=1)).date())
        
        st.markdown("""
        <div class="info-box">
        <strong>Tip:</strong> Choose a broader date range if you're unsure when the e-cheques were received.
        </div>
        """, unsafe_allow_html=True)
        
        submit_button = st.form_submit_button(label="🔍 Search and Download")
    
    if submit_button:
        with st.spinner("Connecting to Gmail and searching for e-Cheques..."):
            try:
                gmail_secrets = config.get('gmail', {})
                progress_container = st.container()
                progress_placeholder = progress_container.empty()
                def progress_callback(message):
                    progress_placeholder.info(message)
                
                downloaded_files, error = gmail_component.search_and_download_echeques(
                    gmail_secrets, 
                    start_date,
                    end_date,
                    progress_callback=progress_callback
                )
                
                if error and not downloaded_files:
                    st.error(f"Error: {error}")
                elif not downloaded_files:
                    st.warning("No e-Cheques found in the date range. Try expanding your search or uploading files manually.")
                else:
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ Successfully downloaded {len(downloaded_files)} e-Cheques!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="subheader">Downloaded e-Cheques</div>', unsafe_allow_html=True)
                    file_data = []
                    for file in downloaded_files:
                        file_data.append({
                            "Filename": file.get('filename', 'Unknown'),
                            "Email Subject": file.get('email_subject', 'Unknown'),
                            "Email Date": file.get('email_date', 'Unknown'),
                            "Size": f"{len(file.get('content', b'')) / 1024:.1f} KB"
                        })
                    
                    file_df = pd.DataFrame(file_data)
                    st.dataframe(file_df, use_container_width=True)
                    
                    st.session_state.downloaded_files = downloaded_files
                    
                    if downloaded_files:
                        st.download_button(
                            label="📥 Download All Attachments as ZIP",
                            data=create_zip_from_files(downloaded_files),
                            file_name="email_attachments.zip",
                            mime="application/zip"
                        )
                    
                    st.markdown("""
                    <div class="info-box">
                    <strong>Next:</strong> Proceed to Step 2 (Process) to extract data from these e-cheques.
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    st.markdown("---")
    st.markdown('<div class="subheader">Or Upload E-Cheques Manually</div>', unsafe_allow_html=True)
    
    st.markdown("""
    If you already have e-cheque PDF files, you can upload them directly instead of downloading from Gmail.
    """)
    
    uploaded_files = st.file_uploader(
        "Drop PDF files here or click to browse", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="Upload one or more PDF files containing e-cheques"
    )
    
    if uploaded_files:
        if st.button("📤 Add Uploaded Files"):
            new_files = []
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.read()
                new_files.append({
                    'filename': uploaded_file.name,
                    'content': file_content,
                    'email_subject': 'Manual Upload',
                    'email_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'size': len(file_content)
                })
            
            st.session_state.downloaded_files.extend(new_files)
            st.markdown(f"""
            <div class="success-box">
            ✅ Added {len(new_files)} files to the download list
            </div>
            """, unsafe_allow_html=True)
            st.rerun()
    
    if st.session_state.downloaded_files and not submit_button:
        st.markdown('<div class="subheader">Files Ready for Processing</div>', unsafe_allow_html=True)
        
        file_data = []
        for file in st.session_state.downloaded_files:
            file_data.append({
                "Filename": file.get('filename', 'Unknown'),
                "Source": file.get('email_subject', 'Manual Upload'),
                "Date": file.get('email_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                "Size": f"{len(file.get('content', b'')) / 1024:.1f} KB"
            })
        
        st.dataframe(file_data, use_container_width=True)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Files"):
                st.session_state.downloaded_files = []
                st.rerun()

# STEP 2: PROCESSING TAB - FULL ORIGINAL CODE RESTORED
with tabs[1]:
    st.markdown('<div class="step-header">Step 2: Process e-Cheques</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This step uses AI to analyze the e-cheque PDFs and extract key information such as:
    - Payee name
    - Amount and currency
    - Cheque date
    - Next actions required
    
    The processed files will be renamed according to a standard format and prepared for upload.
    """)
    
    if not st.session_state.downloaded_files:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Please download e-Cheques from Gmail or upload PDFs directly in Step 1 first.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="subheader">Files Ready for Processing</div>', unsafe_allow_html=True)
        
        skip_processed = st.checkbox("Skip already processed files", value=True,
                                   help="When checked, files that have been previously processed in this session will be skipped.")
        
        files_to_process = []
        skipped_files = []
        
        for file in st.session_state.downloaded_files:
            if skip_processed and file['filename'] in st.session_state.processed_filenames:
                skipped_files.append(file['filename'])
                continue
            files_to_process.append(file)
        
        if skipped_files:
            st.info(f"Skipping {len(skipped_files)} previously processed files")
        
        if not files_to_process:
            st.warning("All files have already been processed. Clear the processed files or uncheck 'Skip already processed files' to reprocess.")
        else:
            file_names = [file['filename'] for file in files_to_process]
            st.markdown(f"**{len(files_to_process)} files ready for processing:**")
            for name in file_names:
                st.markdown(f"- {name}")

            if st.session_state.processed_files:
                st.markdown("---")
                st.markdown('<div class="subheader">Download Previously Processed Files</div>', unsafe_allow_html=True)
                
                zip_files = []
                for processed_file in st.session_state.processed_files:
                    zip_files.append({
                        'filename': processed_file['generated_filename'],
                        'content': processed_file['pdf_data']
                    })
                
                st.download_button(
                    label="📥 Download All Processed Files as ZIP",
                    data=create_zip_from_files(zip_files),
                    file_name=f"all_processed_echeques_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
    
        gemini_api_key = config.get('gemini', {}).get('api_key', '')
        
        if st.button("🔍 Process e-Cheques"):
            if not gemini_api_key:
                st.error("Gemini API key is not configured. Please contact your administrator.")
            elif not files_to_process:
                st.error("No files available to process.")
            else:
                with st.spinner("Processing e-Cheques... This may take a moment."):
                    try:
                        progress_container = st.container()
                        progress_placeholder = progress_container.empty()
                        progress_bar = st.progress(0)
                        
                        def progress_callback(message, progress=None):
                            progress_placeholder.info(message)
                            if progress is not None:
                                progress_bar.progress(progress)
                        
                        processed_files, errors = processing_component.process_echeques(
                            files_to_process, 
                            gemini_api_key,
                            progress_callback=progress_callback
                        )
                        
                        for file in processed_files:
                            if file not in st.session_state.processed_files:
                                st.session_state.processed_files.append(file)
                        
                        for processed_file in processed_files:
                            save_to_db(processed_file)
                        
                        for file in files_to_process:
                            st.session_state.processed_filenames.add(file['filename'])
                        
                        if processed_files:
                            st.markdown(f"""
                            <div class="success-box">
                            ✅ Successfully processed {len(processed_files)} e-cheques!
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('<div class="subheader">Processing Results</div>', unsafe_allow_html=True)
                            
                            results_data = []
                            for result in processed_files:
                                data = result['original_data']
                                results_data.append({
                                    "Original Filename": result.get('original_filename', 'Unknown'),
                                    "Generated Filename": result['generated_filename'],
                                    "Payee": data.get('payee', 'Unknown'),
                                    "Amount": f"{data.get('currency', '')} {data.get('amount_numerical', 'Unknown')}",
                                    "Date": data.get('date', 'Unknown'),
                                    "Teams Folder": data.get('Teams_Folder', 'Not Mapped'),
                                    "GL Code": data.get('GL_Code', 'Not Mapped')
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)

                            zip_files = []
                            for processed_file in processed_files:
                                zip_files.append({
                                    'filename': processed_file['generated_filename'],
                                    'content': processed_file['pdf_data']
                                })
                            
                            st.download_button(
                                label="📥 Download Newly Processed Files as ZIP",
                                data=create_zip_from_files(zip_files),
                                file_name=f"processed_echeques_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
                            
                            st.markdown("""
                            <div class="info-box">
                            <strong>Next:</strong> Proceed to Step 3 (Upload) to send these files to Microsoft Teams.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if errors:
                                with st.expander("Processing Errors"):
                                    for error in errors:
                                        st.error(f"File: {error['filename']} - Error: {error['error']}")
                        else:
                            st.error("No files were successfully processed.")
                            if errors:
                                st.subheader("Errors")
                                for error in errors:
                                    st.error(f"File: {error['filename']} - Error: {error['error']}")
                    
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")

# STEP 3: TEAMS UPLOAD TAB - FULL ORIGINAL CODE RESTORED
with tabs[2]:
    st.markdown('<div class="step-header">Step 3: Upload to Microsoft Teams</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This step uploads the processed e-cheque files to Microsoft Teams in your Finance department.
    Files will be organized into appropriate folders based on their content and properties.
    """)
    
    st.markdown('<div class="subheader">Microsoft Teams Configuration</div>', unsafe_allow_html=True)
    st.info("""
    Microsoft Teams API credentials are configured from secrets.toml:
    - Client ID: ✓ Configured
    - Client Secret: ✓ Configured
    - Tenant ID: ✓ Configured
    - Finance Team ID: ✓ Configured
    """)

    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear All Files", type="primary"):
            st.session_state.downloaded_files = []
            st.session_state.processed_files = []
            st.session_state.processed_filenames = set()
            if 'upload_results' in st.session_state:
                del st.session_state.upload_results
            if 'select_all_files' in st.session_state:
                st.session_state.select_all_files = False
            
            try:
                conn = sqlite3.connect('echeque_processing.db')
                c = conn.cursor()
                c.execute("DELETE FROM processed_files")
                conn.commit()
                conn.close()
                st.success("Successfully cleared all files!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
    with col2:
        st.info("Click 'Clear All Files' to permanently remove all downloaded and processed files from the system.")
    
    if not st.session_state.processed_files:
        st.markdown("""
        <div class="warning-box">
        ⚠️ No processed files to upload. Please complete Step 2 (Process) first.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="subheader">Upload Options</div>', unsafe_allow_html=True)
        
        st.markdown(f"**{len(st.session_state.processed_files)} files available for upload:**")
        
        if 'select_all_files' not in st.session_state:
            st.session_state.select_all_files = False
        
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            if st.button("Select All"):
                st.session_state.select_all_files = True
                st.rerun()
        with col2:
            if st.button("Clear Selection"):
                st.session_state.select_all_files = False
                st.rerun()
        with col3:
            if 'upload_results' in st.session_state and st.button("Reset Upload Status"):
                if 'upload_results' in st.session_state:
                    del st.session_state.upload_results
                st.rerun()
        
        selected_files = []
        with st.container():
            for i, file in enumerate(st.session_state.processed_files):
                if st.checkbox(f"{file['generated_filename']}", value=st.session_state.select_all_files, key=f"pdf_{i}"):
                    selected_files.append(file)
        
        if selected_files:
            st.markdown(f"**{len(selected_files)} files selected for upload**")
        else:
            st.warning("Please select at least one file to upload")
        
        if st.button("📤 Upload to Teams"):
            if not selected_files:
                st.error("Please select at least one file to upload.")
            else:
                with st.spinner(f"Uploading {len(selected_files)} files to Microsoft Teams..."):
                    try:
                        teams_creds = config.get('teams', {})
                        finance_team_id = teams_creds.get('finance_team_id', '')
                        
                        progress_container = st.container()
                        progress_placeholder = progress_container.empty()
                        progress_bar = st.progress(0)
                        
                        def progress_callback(message, progress=None):
                            progress_placeholder.info(message)
                            if progress is not None:
                                progress_bar.progress(progress)
                        
                        if len(selected_files) > 1:
                            progress_placeholder.info(f"Preparing to upload {len(selected_files)} files in batch...")
                            
                        upload_results, error, _, _ = teams_component.upload_files_to_teams(
                            selected_files,
                            teams_creds.get('client_id', ''),
                            teams_creds.get('client_secret', ''),
                            teams_creds.get('tenant_id', ''),
                            finance_team_id,
                            progress_callback=progress_callback
                        )
                        
                        st.session_state.upload_results = upload_results
                        
                        if error:
                            st.error(f"Teams upload failed: {error}")
                        elif upload_results:
                            success_count = sum(1 for result in upload_results if result['success'])
                            
                            if success_count == len(upload_results):
                                st.markdown(f"""
                                <div class="success-box">
                                ✅ Successfully uploaded all {len(upload_results)} files to Teams!
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning(f"Uploaded {success_count} out of {len(upload_results)} files to Teams.")
                            
                            st.markdown('<div class="subheader">Upload Results</div>', unsafe_allow_html=True)
                            
                            results_data = []
                            for result in upload_results:
                                results_data.append({
                                    "Filename": result['filename'],
                                    "Status": "✅ Success" if result['success'] else "❌ Failed",
                                    "Target Folder": result.get('target_folder', 'Unknown'),
                                    "Error": result.get('error', '') if not result['success'] else ''
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            if success_count > 0:
                                st.markdown("""
                                <div class="info-box">
                                <strong>Complete!</strong> The e-cheques have been successfully uploaded to Teams and are now available for the Finance team.
                                </div>
                                """, unsafe_allow_html=True)
                                
                                csv_data = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Upload Report as CSV",
                                    data=csv_data,
                                    file_name=f"teams_upload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                    
                    except Exception as e:
                        st.error(f"An error occurred during Teams upload: {str(e)}")
        
        if 'upload_results' in st.session_state and st.session_state.upload_results:
            st.markdown("---")
            st.markdown('<div class="subheader">Previous Upload Results</div>', unsafe_allow_html=True)
            
            results_data = []
            for result in st.session_state.upload_results:
                results_data.append({
                    "Filename": result['filename'],
                    "Status": "✅ Success" if result['success'] else "❌ Failed",
                    "Target Folder": result.get('target_folder', 'Unknown'),
                    "Error": result.get('error', '') if not result['success'] else ''
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)

# --- NEW: MAPPING MANAGEMENT TAB ---
with tabs[3]:
    st.markdown('<div class="step-header">Manage Payee Mapping Rules</div>', unsafe_allow_html=True)
    st.markdown("""
    Use this spreadsheet editor to manage the rules that map a payee's full name to a specific Teams folder and GL Code.
    - **To Add a Row:** Scroll to the bottom and start typing in the new row.
    - **To Delete a Row:** Select the row(s) using the checkboxes on the left and click the trash can icon 🗑️ that appears.
    - **To Search:** Click the magnifying glass icon 🔍.
    
    **Important: Click the 'Save Changes' button below after you finish editing.**
    """)

    mapping_df = load_mapping_data()

    st.markdown("---")
    
    edited_df = st.data_editor(
        mapping_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Payee": st.column_config.TextColumn("Full Payee Name (from cheque)", required=True),
            "Teams_Folder": st.column_config.TextColumn("Target Teams Folder", required=True),
            "GL_Code": st.column_config.TextColumn("GL Code (Optional)"),
        },
        height=500
    )

    st.markdown("---")

    if st.button("💾 Save Changes to Mapping File", type="primary"):
        if save_mapping_data(edited_df):
            st.success("✅ Mapping rules have been saved successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ There was an error saving the mapping file.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
<p><strong>e-Cheque Processing Pipeline</strong> | Version 1.1.0 | © 2025 WMC Finance Team</p>
<p>For help or support, contact the IT Support Team.</p>
</div>
""", unsafe_allow_html=True)
