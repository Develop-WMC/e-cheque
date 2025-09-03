import base64
import json
import fitz  # PyMuPDF
import os
import re
import pandas as pd
import google.generativeai as genai
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Configuration ---
MAPPING_FILE = "payee_mappings.csv"
MAPPING_COLUMNS = ["Payee", "Teams_Folder"]
MAX_RETRIES = 5
INITIAL_WAIT = 1
MAX_WAIT = 32

TRUST_ACCOUNT_PAYER = "WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT"
WMC_PAYER = "WEALTH MANAGEMENT CUBE LIMITED"

# --- Errors ---
class APIRateLimitError(Exception):
    """Custom exception for API rate limiting errors."""
    pass

# --- Prompt ---
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

# --- Mappings ---
def load_mappings(file_path=MAPPING_FILE):
    """Loads payee mappings from CSV. Ensures expected columns exist."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, dtype=str).fillna("")
        else:
            df = pd.DataFrame(columns=MAPPING_COLUMNS)
        # Ensure required columns present
        for col in MAPPING_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df, None
    except Exception as e:
        # Return empty but shaped DataFrame plus error message
        return pd.DataFrame(columns=MAPPING_COLUMNS), f"Error loading mappings: {str(e)}"

def _standardize(s: str) -> str:
    """Uppercase, strip, and collapse internal whitespace."""
    if s is None:
        return ""
    return " ".join(str(s).upper().strip().split())

def get_mapping_info(payee, mappings_df):
    """Returns Teams_Folder for a given payee, or 'Uncategorized'."""
    if mappings_df.empty or payee is None:
        return "Uncategorized"
    payee_std = _standardize(payee)
    if "Standardized_Name" not in mappings_df.columns:
        mappings_df["Standardized_Name"] = mappings_df["Payee"].astype(str).map(_standardize)
    match = mappings_df[mappings_df["Standardized_Name"] == payee_std]
    if not match.empty:
        folder = str(match.iloc[0].get("Teams_Folder", "")).strip()
        return folder if folder else "Uncategorized"
    return "Uncategorized"

def get_filename_alias(payee: str, payer: str, mappings_df) -> str:
    """
    Old naming system behavior using the new CSV:
    - If payer is the TRUST_ACCOUNT_PAYER: use original payee (no alias).
    - Else: use Teams_Folder as the alias when available; otherwise fall back to original payee.
    """
    if _standardize(payer) == _standardize(TRUST_ACCOUNT_PAYER):
        return payee or ""
    # Use Teams_Folder as "short form" alias when present
    folder = get_mapping_info(payee, mappings_df)
    return folder if folder and folder != "Uncategorized" else (payee or "")

# --- PDF to image ---
def pdf_to_image(pdf_bytes):
    """Converts the first page of a PDF from bytes into a PNG byte string."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if pdf_document.page_count == 0:
            return None, "Uploaded PDF is empty."
        page = pdf_document.load_page(0)
        mat = fitz.Matrix(4, 4)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        pdf_document.close()
        return img_bytes, None
    except Exception as e:
        return None, f"Error converting PDF to image: {str(e)}"

# --- Gemini calls ---
@retry(
    retry=retry_if_exception_type(APIRateLimitError),
    wait=wait_exponential(multiplier=INITIAL_WAIT, max=MAX_WAIT),
    stop=stop_after_attempt(MAX_RETRIES),
    reraise=True
)
def call_gemini_api_with_retry(model, prompt_parts):
    """Calls the Gemini API with an exponential backoff retry mechanism."""
    try:
        response = model.generate_content(prompt_parts)
        if not response:
            raise APIRateLimitError("Empty response from API")
        return response.text.strip()
    except Exception as e:
        msg = str(e)
        if "429" in msg or "resource has been exhausted" in msg.lower():
            raise APIRateLimitError(f"Rate limit exceeded: {msg}")
        raise

def call_gemini_api(image_bytes, prompt, api_key):
    """Configures and calls the Gemini Vision API."""
    if not api_key:
        return None, "Missing Gemini API key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=genai.GenerationConfig(temperature=0.0),
        )
        image_parts = [{"mime_type": "image/png", "data": base64.b64encode(image_bytes).decode("utf-8")}]
        prompt_parts = [prompt, image_parts[0]]

        time.sleep(1)  # polite delay

        try:
            text = call_gemini_api_with_retry(model, prompt_parts)
            return text, None
        except APIRateLimitError as e:
            return None, f"API rate limit error after {MAX_RETRIES} retries: {e}"
        except Exception as e:
            return None, f"Unexpected error during API call: {e}"
    except Exception as e:
        return None, f"Error in API configuration: {str(e)}"

# --- Filename helpers ---
def sanitize_filename(filename):
    """Removes characters that are invalid for filenames."""
    return re.sub(r'[\/*?:"<>|]', "_", str(filename or "")).strip()

def generate_filename(key_identifier, payer, payee_for_filename, currency, is_trailer_fee, is_management_fee):
    """
    Builds the final filename following the old naming system:
    - payee_for_filename has already been mapped (Teams_Folder alias) or is original payee
      per the trust account rule.
    - Suffixes:
        *_T for trailer fees
        * MF for OFS/Oreana management fees
    - Payer-specific patterns:
        WMC:            "{key} WMC-{payee}{suffix}.pdf"
        Trust account:  "{currency} {key} {payee}{suffix}.pdf"
        Other:          "{payee}_{key}_{currency}{suffix}.pdf"
    """
    sanitized_payee = sanitize_filename(payee_for_filename)

    suffix = ""
    if bool(is_trailer_fee):
        suffix = "_T"
    elif bool(is_management_fee) and _standardize(payee_for_filename) in {"OFS", "OREANA FINANCIAL SERVICES LIMITED"}:
        suffix = " MF"

    if payer == WMC_PAYER:
        return f"{key_identifier} WMC-{sanitized_payee}{suffix}.pdf"
    elif payer == TRUST_ACCOUNT_PAYER:
        return f"{currency} {key_identifier} {sanitized_payee}{suffix}.pdf"
    else:
        return f"{sanitized_payee}_{key_identifier}_{currency}{suffix}.pdf"

# --- Core processing ---
def process_echeque(pdf_data, gemini_api_key, mappings_df, custom_prompt=""):
    """Processes a single e-cheque PDF, extracts data, and applies mapping rules."""
    # Convert PDF to image
    image_bytes, error = pdf_to_image(pdf_data)
    if error:
        return None, error

    # Build prompt and call model
    prompt = generate_prompt(custom_prompt)
    raw_response, error = call_gemini_api(image_bytes, prompt, gemini_api_key)
    if error:
        return None, error

    # Parse model response
    try:
        clean_response = raw_response.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(clean_response)

        required_fields = ["date", "payee", "key_identifier", "payer", "currency", "is_trailer_fee", "is_management_fee"]
        missing = [f for f in required_fields if f not in parsed_json]
        if missing:
            return None, f"Missing required fields in API response: {', '.join(missing)}"

        payee_name = parsed_json.get("payee", "")
        payer_name = parsed_json.get("payer", "")

        # Teams_Folder for routing (always compute and attach)
        teams_folder = get_mapping_info(payee_name, mappings_df)
        parsed_json["Teams_Folder"] = teams_folder

        # Old naming system using the new CSV:
        # - Use Teams_Folder as alias unless payer is trust account; then use original payee
        payee_alias_for_filename = get_filename_alias(payee_name, payer_name, mappings_df)

        filename = generate_filename(
            key_identifier=parsed_json["key_identifier"],
            payer=payer_name,
            payee_for_filename=payee_alias_for_filename,
            currency=parsed_json["currency"],
            is_trailer_fee=parsed_json.get("is_trailer_fee", False),
            is_management_fee=parsed_json.get("is_management_fee", False),
        )

        return {
            "original_data": parsed_json,
            "generated_filename": filename,
            "pdf_data": pdf_data,
        }, None

    except json.JSONDecodeError as e:
        sample = raw_response[:500] if isinstance(raw_response, str) else str(raw_response)[:500]
        return None, f"Error parsing JSON response: {e}. Response was: {sample}"
    except Exception as e:
        return None, f"Error processing e-cheque: {e}"

def process_echeques(downloaded_files, gemini_api_key, progress_callback=None):
    """Processes a batch of e-cheque files."""
    processed_files = []
    errors = []

    mappings_df, mapping_error = load_mappings()
    if mapping_error:
        print(f"WARNING: Could not load mapping file. {mapping_error}")

    total_files = len(downloaded_files)
    for i, file_info in enumerate(downloaded_files):
        try:
            # Progress notification (supports both 1-arg and 2-arg callbacks)
            if progress_callback:
                msg = f"Processing file {i + 1}/{total_files}: {file_info.get('filename', 'Unknown')}"
                progress = (i + 1) / total_files if total_files else 1.0
                try:
                    progress_callback(msg, progress)
                except TypeError:
                    progress_callback(msg)

            if i > 0:
                time.sleep(2)  # gentle throttling between files

            result, error = process_echeque(file_info["content"], gemini_api_key, mappings_df)
            if error:
                errors.append({"filename": file_info.get("filename", "Unknown"), "error": error})
                continue

            result["original_filename"] = file_info.get("filename", "Unknown")
            processed_files.append(result)

        except Exception as e:
            errors.append({"filename": file_info.get("filename", "Unknown"), "error": f"An unexpected error occurred: {e}"})

    return processed_files, errors
