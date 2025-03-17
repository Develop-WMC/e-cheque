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

# Constants
MAPPING_FILE = "payee_mappings.csv"
MAPPING_COLUMNS = ['Full Name', 'Short Form']
MAX_RETRIES = 5
INITIAL_WAIT = 1  # seconds
MAX_WAIT = 32  # seconds

class APIRateLimitError(Exception):
    pass

def generate_prompt(override_prompt: str = "") -> str:
    if override_prompt:
        return override_prompt

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

def load_mappings(file_path=MAPPING_FILE):
    """Load payee mappings from CSV file"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=MAPPING_COLUMNS)
        return df, None
    except Exception as e:
        return None, f"Error loading mappings: {str(e)}"

def save_mappings(df, file_path=MAPPING_FILE):
    """Save payee mappings to CSV file"""
    try:
        df.to_csv(file_path, index=False)
        return True, None
    except Exception as e:
        return False, f"Error saving mappings: {str(e)}"

def get_payee_shortform(payee, mappings_df):
    """Get short form of payee name from mappings"""
    if mappings_df.empty:
        return payee
        
    payee_upper = payee.upper().strip()
    # Remove extra spaces between words and standardize spaces
    payee_upper = ' '.join(payee_upper.split())
    
    # Do the same standardization for the mapping names
    mappings_df['Standardized_Name'] = mappings_df['Full Name'].str.upper().str.strip().apply(lambda x: ' '.join(x.split()))
    
    match = mappings_df[mappings_df['Standardized_Name'] == payee_upper]
    if not match.empty:
        return match.iloc[0]['Short Form']
    return payee

def pdf_to_image(pdf_bytes):
    """Convert PDF to image"""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if pdf_document.page_count == 0:
            return None, "Uploaded PDF is empty."

        page = pdf_document.load_page(0)
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        pdf_document.close()
        return img_bytes, None
    except Exception as e:
        return None, f"Error converting PDF to image: {str(e)}"

def is_rate_limit_error(exception):
    return isinstance(exception, APIRateLimitError) or (
        isinstance(exception, Exception) and 
        "429" in str(exception)
    )

@retry(
    retry=retry_if_exception_type(APIRateLimitError),
    wait=wait_exponential(multiplier=INITIAL_WAIT, max=MAX_WAIT),
    stop=stop_after_attempt(MAX_RETRIES),
    reraise=True
)
def call_gemini_api_with_retry(model, prompt_parts):
    try:
        response = model.generate_content(prompt_parts)
        if not response:
            raise APIRateLimitError("Empty response from API")
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            raise APIRateLimitError(f"Rate limit exceeded: {str(e)}")
        raise e

def call_gemini_api(image_bytes, prompt, api_key):
    """Call Gemini Vision API to analyze e-cheque"""
    if not api_key:
        return None, "Missing Gemini API key."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash', 
                                    generation_config=genai.GenerationConfig(temperature=0.0))

        image_parts = [{"mime_type": "image/png", 
                       "data": base64.b64encode(image_bytes).decode("utf-8")}]
        prompt_parts = [prompt, image_parts[0]]
        
        # Add delay between requests
        time.sleep(1)  # Add 1 second delay between requests
        
        try:
            response_text = call_gemini_api_with_retry(model, prompt_parts)
            return response_text, None
        except APIRateLimitError as e:
            return None, (f"Rate limit error after {MAX_RETRIES} retries. "
                        f"Last error: {str(e)}. "
                        f"Please wait a few minutes before trying again.")
        except Exception as e:
            return None, f"Unexpected error during API call: {str(e)}"
            
    except Exception as e:
        return None, f"Error in API configuration: {str(e)}"

def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    invalid_chars = r'[\/*?:"<>|]'
    return re.sub(invalid_chars, '_', filename)

def generate_filename(key_identifier, payer, payee, currency, is_trailer_fee, is_management_fee):
    """Generate appropriate filename based on extracted data"""
    sanitized_payee = sanitize_filename(payee)

    # Check for trailer fee using AI's judgment
    if is_trailer_fee:
        if payer == "WEALTH MANAGEMENT CUBE LIMITED":
            return f"{key_identifier} WMC-{sanitized_payee}_T.pdf"
        elif payer == "WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT":
            return f"{currency} {key_identifier} {sanitized_payee}_T.pdf"
        else:
            return f"{sanitized_payee}_{key_identifier}_{currency}_T.pdf"
    
    # Check for management fee using AI's judgment
    elif is_management_fee and payee.upper() in ['OFS', 'OREANA FINANCIAL SERVICES LIMITED']:
        if payer == "WEALTH MANAGEMENT CUBE LIMITED":
            return f"{key_identifier} WMC-{sanitized_payee} MF.pdf"
        elif payer == "WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT":
            return f"{currency} {key_identifier} {sanitized_payee} MF.pdf"
        else:
            return f"{sanitized_payee}_{key_identifier}_{currency} MF.pdf"
    
    # Default naming without special suffixes
    else:
        if payer == "WEALTH MANAGEMENT CUBE LIMITED":
            return f"{key_identifier} WMC-{sanitized_payee}.pdf"
        elif payer == "WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT":
            return f"{currency} {key_identifier} {sanitized_payee}.pdf"
        else:
            return f"{sanitized_payee}_{key_identifier}_{currency}.pdf"

def process_echeque(pdf_data, gemini_api_key, mappings_df=None, custom_prompt=""):
    """Process a single e-cheque file"""
    # If no mappings provided, try to load
    if mappings_df is None:
        mappings_df, error = load_mappings()
        if error:
            mappings_df = pd.DataFrame(columns=MAPPING_COLUMNS)
    
    # Convert PDF to image
    image_bytes, error = pdf_to_image(pdf_data)
    if error:
        return None, error
    
    # Get prompt
    prompt = generate_prompt(custom_prompt)
    
    # Call Gemini API
    raw_response, error = call_gemini_api(image_bytes, prompt, gemini_api_key)
    if error:
        return None, error
    
    # Process response
    try:
        # Clean the response string
        clean_response = raw_response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:-3]
        
        parsed_json = json.loads(clean_response)
        
        # Check for required fields
        required_fields = ["date", "payee", "key_identifier", "payer", "currency", "is_trailer_fee", "is_management_fee"]
        if not all(field in parsed_json for field in required_fields):
            missing = [field for field in required_fields if field not in parsed_json]
            return None, f"Missing required fields in API response: {', '.join(missing)}"
        
        # Get short form of payee
        original_payee = parsed_json['payee']
        
        # Only apply mapping if payer is NOT "WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT"
        if parsed_json['payer'] != "WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT":
            shortened_payee = get_payee_shortform(original_payee, mappings_df)
        else:
            # For client trust account, use the original payee name
            shortened_payee = original_payee
        
        # Generate filename
        filename = generate_filename(
            key_identifier=parsed_json['key_identifier'],
            payer=parsed_json['payer'],
            payee=shortened_payee,
            currency=parsed_json['currency'],
            is_trailer_fee=parsed_json['is_trailer_fee'],
            is_management_fee=parsed_json['is_management_fee']
        )
        
        # Return results
        return {
            'original_data': parsed_json,
            'original_payee': original_payee,
            'mapped_payee': shortened_payee,
            'generated_filename': filename,
            'pdf_data': pdf_data,
            'next_step': parsed_json.get('next_step', 'Unknown')
        }, None
        
    except json.JSONDecodeError as e:
        return None, f"Error parsing JSON response: {str(e)}"
    except Exception as e:
        return None, f"Error processing e-cheque: {str(e)}"

def process_echeques(downloaded_files, gemini_api_key, progress_callback=None):
    """Process multiple e-cheque files"""
    processed_files = []
    errors = []
    
    # Load mappings once for all files
    mappings_df, error = load_mappings()
    if error:
        mappings_df = pd.DataFrame(columns=MAPPING_COLUMNS)
    
    total_files = len(downloaded_files)
    for i, file_info in enumerate(downloaded_files):
        try:
            if progress_callback:
                progress_callback(f"Processing file {i+1}/{total_files}: {file_info['filename']}")
            
            # Add delay between files
            if i > 0:
                time.sleep(2)  # Add 2 second delay between files
            
            # Process each file
            result, error = process_echeque(file_info['content'], gemini_api_key, mappings_df)
            
            if error:
                errors.append({
                    'filename': file_info['filename'],
                    'error': error
                })
                continue
            
            # Add original file info to result
            result['original_filename'] = file_info['filename']
            result['email_subject'] = file_info.get('email_subject', 'Unknown')
            result['email_date'] = file_info.get('email_date', 'Unknown')
            
            processed_files.append(result)
            
        except Exception as e:
            errors.append({
                'filename': file_info['filename'],
                'error': f"Unexpected error: {str(e)}"
            })
            
    return processed_files, errors
