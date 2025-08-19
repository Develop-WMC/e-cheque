import base64
import json
import io
import fitz  # PyMuPDF
from PIL import Image
import os
import re
from datetime import datetime
import pandas as pd
import csv
import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- MODIFIED: Standardized mapping file name and columns to match app.py ---
MAPPING_FILE = "payee_mapping.csv"
MAPPING_COLUMNS = ['Payee', 'Teams_Folder', 'GL_Code']
MAX_RETRIES = 5
INITIAL_WAIT = 1
MAX_WAIT = 32

class APIRateLimitError(Exception):
    """Custom exception for API rate limiting errors."""
    pass

def generate_prompt(override_prompt: str = "") -> str:
    """Generates the detailed prompt for the AI model."""
    if override_prompt:
        return override_prompt
    # Using your original, detailed prompt
    prompt = """
    Extract the following information from this e-cheque and return it as JSON. For the currency field, 
    please normalize it according to these rules:
    - '¥' or '￥' or 'RMB' should be normalized to 'CNY'
    - '$' or 'USD' or 'US$' should be normalized to 'USD'
    - 'HK$' or 'HKD' should be normalized to 'HKD'
    - '€' should be normalized to 'EUR'
    - '£' should be normalized to 'GBP'

    Also, analyze the remarks field to determine if this is:
    1. A trailer fee payment (includes any mention of trailer, rebate for trailer, etc.)
    2. A management fee payment (only for OFS/Oreana Financial Services, includes managed services fee, management fee, etc.)

    Schema:
    {
      "type": "object",
      "properties": {
        "bank_name": { "type": "string", "description": "The name of the bank issuing the e-cheque." },
        "date": { "type": "string", "format": "date", "description": "The date the e-cheque was issued (YYYY-MM-DD)." },
        "payee": { "type": "string", "description": "The name of the person or entity to whom the e-cheque is payable." },
        "payer": { "type": "string", "description": "The name of the account the funds are drawn from." },
        "amount_numerical": { "type": "string", "description": "The amount of the e-cheque in numerical form (e.g., 66969.77)." },
        "amount_words": { "type": "string", "description": "The amount of the e-cheque in words." },
        "cheque_number": { "type": "string", "description": "The full cheque number, including all digits and spaces." },
        "key_identifier": { "type": "string", "description": "The first six digits of the cheque number." },
        "currency": { "type": "string", "description": "The normalized currency code (CNY, USD, HKD, EUR, GBP)"},
        "remarks": { "type": "string", "description": "The remark of the e-cheque"},
        "is_trailer_fee": { "type": "boolean", "description": "True if this is a trailer fee payment based on remarks" },
        "is_management_fee": { "type": "boolean", "description": "True if this is a management fee payment for OFS/Oreana" },
        "next_step": { "type": "string" }
      },
      "required": ["date", "payee", "amount_numerical", "key_identifier", "payer", "next_step", "is_trailer_fee", "is_management_fee"]
    }

    Rules for next_step determination:
    1. If the 'remarks' field contains "URGENT", set 'next_step' to 'Flag for Manual Review'
    2. If the 'currency' is not 'HKD', set 'next_step' to 'Flag for Manual Review'
    3. Otherwise, set 'next_step' to 'Process Payment'

    Return only the JSON object with no additional text or formatting.
    """
    return prompt

# --- NEW: Function to load the mapping rules from the CSV ---
def load_mappings(file_path=MAPPING_FILE):
    """Loads payee mappings from the CSV file into a pandas DataFrame."""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path), None
        else:
            # If the file doesn't exist, return an empty DataFrame to avoid errors
            return pd.DataFrame(columns=MAPPING_COLUMNS), None
    except Exception as e:
        # Return an empty DataFrame and the error message if loading fails
        return pd.DataFrame(columns=MAPPING_COLUMNS), f"Error loading mappings: {str(e)}"

# --- NEW: Function to find the mapping info for a specific payee ---
def get_mapping_info(payee, mappings_df):
    """
    Looks up the Teams Folder and GL Code for a given payee name from the mappings DataFrame.
    Performs a case-insensitive and whitespace-insensitive comparison.
    """
    if mappings_df.empty or payee is None:
        return 'Uncategorized', 'N/A' # Default values if no mapping file or payee

    # Standardize the input payee name for a reliable match
    payee_upper = str(payee).upper().strip()
    payee_upper = ' '.join(payee_upper.split())
    
    # Create a standardized column in the DataFrame for comparison
    mappings_df['Standardized_Name'] = mappings_df['Payee'].astype(str).str.upper().str.strip().apply(lambda x: ' '.join(x.split()) if pd.notna(x) else '')
    
    # Find the matching row
    match = mappings_df[mappings_df['Standardized_Name'] == payee_upper]
    
    if not match.empty:
        # If a match is found, return the folder and GL code
        folder = match.iloc[0]['Teams_Folder']
        gl_code = match.iloc[0]['GL_Code']
        return folder, gl_code
        
    # If no match is found, return default values
    return 'Uncategorized', 'N/A'

def pdf_to_image(pdf_bytes):
    """Converts the first page of a PDF from bytes into an image in bytes."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if pdf_document.page_count == 0: return None, "Uploaded PDF is empty."
        page = pdf_document.load_page(0); zoom = 4; mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png"); pdf_document.close()
        return img_bytes, None
    except Exception as e: return None, f"Error converting PDF to image: {str(e)}"

@retry(retry=retry_if_exception_type(APIRateLimitError), wait=wait_exponential(multiplier=INITIAL_WAIT, max=MAX_WAIT), stop=stop_after_attempt(MAX_RETRIES), reraise=True)
def call_gemini_api_with_retry(model, prompt_parts):
    """Calls the Gemini API with an exponential backoff retry mechanism."""
    try:
        response = model.generate_content(prompt_parts)
        if not response: raise APIRateLimitError("Empty response from API")
        return response.text.strip()
    except Exception as e:
        if "429" in str(e) or "resource has been exhausted" in str(e).lower():
            raise APIRateLimitError(f"Rate limit exceeded: {str(e)}")
        raise e

def call_gemini_api(image_bytes, prompt, api_key):
    """Configures and calls the Gemini Vision API."""
    if not api_key: return None, "Missing Gemini API key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash', generation_config=genai.GenerationConfig(temperature=0.0))
        image_parts = [{"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("utf-8")}]
        prompt_parts = [prompt, image_parts[0]]; time.sleep(1) # Small delay to respect rate limits
        try:
            return call_gemini_api_with_retry(model, prompt_parts), None
        except APIRateLimitError as e: return None, f"API rate limit error after {MAX_RETRIES} retries: {e}"
        except Exception as e: return None, f"Unexpected error during API call: {e}"
    except Exception as e: return None, f"Error in API configuration: {str(e)}"

def sanitize_filename(filename):
    """Removes characters that are invalid for filenames."""
    return re.sub(r'[\/*?:"<>|]', '_', str(filename))

def generate_filename(key_identifier, payer, payee, currency, is_trailer_fee, is_management_fee):
    """Generates a standardized filename based on extracted e-cheque data."""
    sanitized_payee = sanitize_filename(payee)
    suffix = ""
    if is_trailer_fee: suffix = "_T"
    elif is_management_fee and str(payee).upper() in ['OFS', 'OREANA FINANCIAL SERVICES LIMITED']: suffix = " MF"

    if payer == "WEALTH MANAGEMENT CUBE LIMITED": return f"{key_identifier} WMC-{sanitized_payee}{suffix}.pdf"
    elif payer == "WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT": return f"{currency} {key_identifier} {sanitized_payee}{suffix}.pdf"
    else: return f"{sanitized_payee}_{key_identifier}_{currency}{suffix}.pdf"

# --- MODIFIED: This function now accepts the mappings_df to perform the lookup ---
def process_echeque(pdf_data, gemini_api_key, mappings_df, custom_prompt=""):
    """Processes a single e-cheque PDF, extracts data, and applies mapping rules."""
    image_bytes, error = pdf_to_image(pdf_data)
    if error: return None, error
    
    prompt = generate_prompt(custom_prompt)
    raw_response, error = call_gemini_api(image_bytes, prompt, gemini_api_key)
    if error: return None, error
    
    try:
        clean_response = raw_response.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(clean_response)
        
        required_fields = ["date", "payee", "key_identifier", "payer", "currency", "is_trailer_fee", "is_management_fee"]
        if not all(field in parsed_json for field in required_fields):
            missing = [f for f in required_fields if f not in parsed_json]
            return None, f"Missing required fields in API response: {', '.join(missing)}"
        
        payee_name = parsed_json.get('payee')
        
        # --- CORE LOGIC: Use the mapping file to get folder and GL code ---
        teams_folder, gl_code = get_mapping_info(payee_name, mappings_df)
        parsed_json['Teams_Folder'] = teams_folder
        parsed_json['GL_Code'] = gl_code
        # --- END OF CORE LOGIC ---
        
        filename = generate_filename(
            key_identifier=parsed_json['key_identifier'],
            payer=parsed_json['payer'],
            payee=payee_name,
            currency=parsed_json['currency'],
            is_trailer_fee=parsed_json.get('is_trailer_fee', False),
            is_management_fee=parsed_json.get('is_management_fee', False)
        )
        
        return {'original_data': parsed_json, 'generated_filename': filename, 'pdf_data': pdf_data}, None
        
    except json.JSONDecodeError as e: return None, f"Error parsing JSON response: {e}. Response was: {raw_response[:500]}"
    except Exception as e: return None, f"Error processing e-cheque: {e}"

# --- MODIFIED: This function now loads the mapping file once before processing ---
def process_echeques(downloaded_files, gemini_api_key, progress_callback=None):
    """
    Processes a batch of e-cheque files.
    It loads the payee mapping rules once for efficiency.
    """
    processed_files = []; errors = []
    
    # Load mappings once for the entire batch to improve performance
    mappings_df, mapping_error = load_mappings()
    if mapping_error:
        # Log the warning but continue processing with default values
        print(f"WARNING: Could not load mapping file. {mapping_error}")
    
    total_files = len(downloaded_files)
    for i, file_info in enumerate(downloaded_files):
        try:
            if progress_callback:
                progress_callback(f"Processing file {i+1}/{total_files}: {file_info['filename']}", (i + 1) / total_files)
            
            # Add a delay between API calls to avoid rate limiting
            if i > 0: time.sleep(2) 
            
            # Pass the loaded mappings_df to the single-file processor
            result, error = process_echeque(file_info['content'], gemini_api_key, mappings_df)
            
            if error:
                errors.append({'filename': file_info['filename'], 'error': error})
                continue
            
            result['original_filename'] = file_info['filename']
            processed_files.append(result)
            
        except Exception as e:
            errors.append({'filename': file_info['filename'], 'error': f"An unexpected error occurred: {e}"})
            
    return processed_files, errors
