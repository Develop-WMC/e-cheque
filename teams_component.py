import msal
import requests
import time
import os
import re
import urllib.parse
import json
import random
import string
import base64
from datetime import datetime

def sanitize_filename(filename):
    """Remove or replace invalid characters for SharePoint/Teams filenames"""
    # Characters not allowed in SharePoint: " * : < > ? / \ | # % { } ~
    invalid_chars = r'[\"*:<>?/\\|#%{}~]'
    # Replace invalid characters with underscores
    sanitized = re.sub(invalid_chars, '_', filename)
    # Remove leading and trailing spaces
    sanitized = sanitized.strip()
    # Remove leading dots (SharePoint doesn't allow filenames starting with dots)
    sanitized = re.sub(r'^\.+', '', sanitized)
    # Limit filename length to 240 characters (to be safe)
    if len(sanitized) > 240:
        name_part, ext_part = os.path.splitext(sanitized)
        sanitized = name_part[:240-len(ext_part)] + ext_part
    return sanitized

def get_random_suffix():
    """Generate a random string to make filenames unique"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{random_str}"

def ensure_valid_token(client_id, client_secret, tenant_id, current_token=None, token_expires_at=0):
    """Ensure we have a valid token before making API calls"""
    current_time = time.time()
    
    # If token is missing or about to expire (or has expired), get a new one
    if (not current_token or token_expires_at < current_time + 300):
        authority = f"https://login.microsoftonline.com/{tenant_id}"
        
        app = msal.ConfidentialClientApplication(
            client_id,
            authority=authority,
            client_credential=client_secret
        )
        
        # Get a new token
        result = app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )
        
        if "access_token" in result:
            new_token = result["access_token"]
            new_expires_at = current_time + result.get("expires_in", 3600)
            return new_token, new_expires_at, app, None
        elif "error" in result:
            error_msg = f"Authentication error: {result.get('error')} - {result.get('error_description')}"
            return None, 0, None, error_msg
    
    return current_token, token_expires_at, None, None

def get_teams(access_token):
    """Get all teams in the organization"""
    try:
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Try multiple endpoints to get teams data
        endpoints = [
            'https://graph.microsoft.com/v1.0/teams',
            'https://graph.microsoft.com/v1.0/groups?$filter=resourceProvisioningOptions/Any(x:x eq \'Team\')',
            'https://graph.microsoft.com/v1.0/groups?$select=id,displayName,description'
        ]
        
        for endpoint in endpoints:
            response = requests.get(endpoint, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'value' in data and len(data['value']) > 0:
                    return data['value'], None
        
        # If we're here, all methods failed
        return None, "Failed to retrieve teams data from all endpoints"
    except Exception as e:
        return None, f"Failed to get teams: {str(e)}"

def get_team_drive_folders(access_token, team_id, parent_folder_id='root'):
    """Get folders in a team drive"""
    try:
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Get the team's SharePoint site
        response = requests.get(
            f'https://graph.microsoft.com/v1.0/groups/{team_id}/sites/root',
            headers=headers
        )
        
        if response.status_code != 200:
            return None, None, f"Error getting team site: {response.status_code}"
        
        site_id = response.json()['id']
        
        # Get the drives in the site
        response = requests.get(
            f'https://graph.microsoft.com/v1.0/sites/{site_id}/drives',
            headers=headers
        )
        
        if response.status_code != 200:
            return None, None, f"Error getting site drives: {response.status_code}"
        
        # Get the documents drive (usually the first one)
        drives = response.json()['value']
        if not drives:
            return None, None, "No drives found in the team site"
        
        drive_id = drives[0]['id']
        
        # Get items in the parent folder
        response = requests.get(
            f'https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{parent_folder_id}/children',
            headers=headers
        )
        
        if response.status_code != 200:
            return drive_id, None, f"Error getting folder items: {response.status_code}"
        
        # Sort items - folders first, then files
        items = response.json()['value']
        folders = [item for item in items if 'folder' in item]
        files = [item for item in items if 'folder' not in item]
        
        # Sort alphabetically
        folders.sort(key=lambda x: x['name'].lower())
        files.sort(key=lambda x: x['name'].lower())
        
        all_items = folders + files
        
        return drive_id, all_items, None
    except Exception as e:
        return None, None, f"Failed to get team drive folders: {str(e)}"

def determine_target_folder(filename, finance_team_id, access_token):
    """Determine which folder to upload the file to based on filename pattern and return full folder path"""
    # Pattern matching for Type 1: "000495 WMC-AAM.pdf"
    # Files from WEALTH MANAGEMENT CUBE LIMITED
    if re.match(r'^\d+ WMC-.*\.pdf$', filename):
        folder_id = "01OU6MNL3KE3XP2T5JMZC244U33CGKOAMH"  # WMC E-cheque folder ID
        folder_path = "Finance Staff/Bank/Cashflow/WMC E-cheque"
        return folder_id, folder_path, "WMC E-cheque"
    
    # Pattern matching for Type 2: "HKD 100671 Cheung Wilma Veronica.pdf"
    # Files from WMC NOMINEE LIMITED-CLIENT TRUST ACCOUNT
    elif re.match(r'^[A-Z]{3} \d+ .*\.pdf$', filename):
        folder_id = "01OU6MNL5F3ZCFDTEACRAJDOX2G4NUMX72"  # E cheque WMC Nominee folder ID
        folder_path = "Finance Staff/Bank/E cheque WMC Nominee"
        return folder_id, folder_path, "E cheque WMC Nominee"
    
    # Default to WMC E-cheque folder for any unmatched patterns
    folder_id = "01OU6MNL3KE3XP2T5JMZC244U33CGKOAMH"  # WMC E-cheque folder ID
    folder_path = "Finance Staff/Bank/Cashflow/WMC E-cheque"
    return folder_id, folder_path, "WMC E-cheque"

def upload_with_sharepoint_api(access_token, finance_team_id, folder_path, file_data, filename, folder_id=None, client_id=None, client_secret=None, tenant_id=None, progress_callback=None):
    """Upload file using SharePoint REST API directly with improved handling for existing files"""
    try:
        # Step 1: Get site ID
        if progress_callback:
            progress_callback("Getting SharePoint site information...")
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json;odata.metadata=minimal'
        }
        
        # Get the team's SharePoint site
        graph_site_url = f'https://graph.microsoft.com/v1.0/groups/{finance_team_id}/sites/root'
        response = requests.get(graph_site_url, headers=headers)
        
        if response.status_code != 200:
            error_detail = ""
            try:
                error_detail = json.dumps(response.json())
            except:
                error_detail = response.text
            return False, f"Error getting SharePoint site: {response.status_code} - {error_detail}"
        
        site_info = response.json()
        site_url = site_info.get('webUrl', '')
        site_id = site_info.get('id')
        
        if not site_url or not site_id:
            return False, "Error: Could not determine SharePoint site URL or ID"
        
        if progress_callback:
            progress_callback(f"Found SharePoint site: {site_url}")
        
        # Get the drive ID for the site
        drive_response = requests.get(
            f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives",
            headers=headers
        )
        
        if drive_response.status_code != 200:
            error_detail = ""
            try:
                error_detail = json.dumps(drive_response.json())
            except:
                error_detail = drive_response.text
            return False, f"Failed to get drives: {drive_response.status_code} - {error_detail}"
        
        drives = drive_response.json().get('value', [])
        if not drives:
            return False, "No drives found for site"
        
        drive_id = drives[0]['id']
        
        if progress_callback:
            progress_callback(f"Found drive ID: {drive_id}")
        
        # Sanitize filename but don't make it unique (allowing overwrites)
        safe_filename = sanitize_filename(filename)
        if safe_filename != filename and progress_callback:
            progress_callback(f"Filename sanitized: '{filename}' → '{safe_filename}'")
        
        # Just use the sanitized filename (enabling overwrites)
        file_to_upload = safe_filename
        
        # IMPORTANT: Check if file already exists
        check_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children?$filter=name eq '{file_to_upload}'"
        check_headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        check_response = requests.get(check_url, headers=check_headers)
        
        existing_file_id = None
        if check_response.status_code == 200:
            items = check_response.json().get('value', [])
            if items:
                existing_file_id = items[0]['id']
                if progress_callback:
                    progress_callback(f"File already exists with ID: {existing_file_id}, will update content")
        
        # For small files (less than 4MB), we can do a simple direct upload
        if len(file_data) < 4 * 1024 * 1024:
            if progress_callback:
                progress_callback("File is small enough for direct upload")
            
            if existing_file_id:
                # Update existing file content
                upload_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{existing_file_id}/content"
                upload_headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/octet-stream'
                }
                
                upload_response = requests.put(upload_url, headers=upload_headers, data=file_data)
                
                if upload_response.status_code in [200, 201]:
                    if progress_callback:
                        progress_callback("Updated existing file successfully!")
                    return True, None
                
                # Log the error for debugging
                error_detail = ""
                try:
                    error_detail = json.dumps(upload_response.json())
                except:
                    error_detail = upload_response.text
                
                return False, f"Failed to update content: {upload_response.status_code} - {error_detail}"
            else:
                # Create new file with content
                upload_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}:/{file_to_upload}:/content"
                upload_headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/octet-stream'
                }
                
                upload_response = requests.put(upload_url, headers=upload_headers, data=file_data)
                
                if upload_response.status_code in [200, 201]:
                    if progress_callback:
                        progress_callback("Direct upload successful!")
                    return True, None
                
                # Log the error for debugging
                error_detail = ""
                try:
                    error_detail = json.dumps(upload_response.json())
                except:
                    error_detail = upload_response.text
                
                return False, f"Failed to upload content: {upload_response.status_code} - {error_detail}"
        
        # For larger files, use upload session
        if progress_callback:
            progress_callback("Creating upload session...")
        
        if existing_file_id:
            # Create upload session for existing file
            session_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{existing_file_id}/createUploadSession"
        else:
            # Create upload session for new file
            session_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}:/{file_to_upload}:/createUploadSession"
        
        session_headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Include conflictBehavior in the request body
        session_payload = {
            "item": {
                "@microsoft.graph.conflictBehavior": "replace"
            }
        }
        
        session_response = requests.post(session_url, headers=session_headers, json=session_payload)
        
        if session_response.status_code != 200:
            error_detail = ""
            try:
                error_detail = json.dumps(session_response.json())
            except:
                error_detail = session_response.text
            
            return False, f"Failed to create upload session: {session_response.status_code} - {error_detail}"
        
        upload_url = session_response.json().get('uploadUrl')
        
        if not upload_url:
            return False, "No upload URL returned in session response"
        
        # Upload in chunks
        chunk_size = 1 * 1024 * 1024  # 1MB chunks
        total_size = len(file_data)
        uploaded = 0
        
        while uploaded < total_size:
            chunk_data = file_data[uploaded:uploaded + chunk_size]
            chunk_length = len(chunk_data)
            
            start = uploaded
            end = uploaded + chunk_length - 1
            
            chunk_headers = {
                'Content-Length': str(chunk_length),
                'Content-Range': f'bytes {start}-{end}/{total_size}'
            }
            
            chunk_response = requests.put(
                upload_url,
                headers=chunk_headers,
                data=chunk_data
            )
            
            if chunk_response.status_code not in [200, 201, 202]:
                error_detail = ""
                try:
                    error_detail = json.dumps(chunk_response.json())
                except:
                    error_detail = chunk_response.text
                
                return False, f"Error uploading chunk {start}-{end}: {chunk_response.status_code} - {error_detail}"
            
            uploaded += chunk_length
            if progress_callback:
                progress_callback(f"Uploaded {uploaded/total_size:.1%} ({uploaded}/{total_size} bytes)")
            
            if chunk_response.status_code in [200, 201]:
                if progress_callback:
                    progress_callback("Chunked upload successful!")
                return True, None
        
        return False, "Upload completed but unexpected response received"
    except Exception as e:
        return False, f"SharePoint upload error: {str(e)}"

def upload_file_legacy(access_token, drive_id, folder_id, file_data, filename, progress_callback=None):
    """Original upload method - retained for reference but no longer used"""
    try:
        # Sanitize the filename to ensure it's compatible with SharePoint
        safe_filename = sanitize_filename(filename)
        
        # If the filename was changed, log this information
        if safe_filename != filename and progress_callback:
            progress_callback(f"Filename sanitized: '{filename}' → '{safe_filename}'")
        
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        file_size = len(file_data)
        
        # For files smaller than 4MB, we can do a simple upload
        if file_size < 4 * 1024 * 1024:
            content_type = 'application/octet-stream'
            
            # Upload the file
            upload_headers = headers.copy()
            upload_headers['Content-Type'] = content_type
            
            upload_url = f'https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}:/{safe_filename}:/content'
            response = requests.put(
                upload_url,
                headers=upload_headers,
                data=file_data
            )
            
            if response.status_code in [200, 201]:
                return True, None
            else:
                error_detail = ""
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                return False, f"Error uploading {safe_filename}: {response.status_code} - {error_detail}"
        else:
            # For larger files, use upload session
            # Create an upload session
            create_session_url = f'https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}:/{safe_filename}:/createUploadSession'
            response = requests.post(
                create_session_url,
                headers={
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                },
                json={}
            )
            
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                return False, f"Error creating upload session: {response.status_code} - {error_detail}"
            
            upload_url = response.json()['uploadUrl']
            
            # Upload the file in chunks
            chunk_size = 3 * 1024 * 1024  # 3MB chunks
            total_size = file_size
            uploaded = 0
            
            file_content_bytes = file_data
            while uploaded < total_size:
                chunk_data = file_content_bytes[uploaded:uploaded + chunk_size]
                chunk_length = len(chunk_data)
                
                start = uploaded
                end = uploaded + chunk_length - 1
                
                chunk_headers = {
                    'Content-Length': str(chunk_length),
                    'Content-Range': f'bytes {start}-{end}/{total_size}'
                }
                
                chunk_response = requests.put(
                    upload_url,
                    headers=chunk_headers,
                    data=chunk_data
                )
                
                if chunk_response.status_code not in [200, 201, 202]:
                    error_detail = ""
                    try:
                        error_detail = chunk_response.json()
                    except:
                        error_detail = chunk_response.text
                    return False, f"Error uploading chunk {start}-{end}: {chunk_response.status_code} - {error_detail}"
                
                uploaded += chunk_length
                if progress_callback:
                    progress_callback(uploaded / total_size)
                
                if chunk_response.status_code in [200, 201]:
                    return True, None
            
            return False, "Upload failed - Unknown error"
    except Exception as e:
        return False, f"Upload error: {str(e)}"

def upload_file(access_token, drive_id, folder_id, file_data, filename, finance_team_id=None, folder_path=None, client_id=None, client_secret=None, tenant_id=None, progress_callback=None):
    """Upload a file to the specified folder - simplified to always use folder ID"""
    try:
        # Always use the direct SharePoint API method with folder ID
        if progress_callback:
            progress_callback(f"Uploading {filename} to folder ID: {folder_id}")
        
        return upload_with_sharepoint_api(
            access_token,
            finance_team_id,
            folder_path,
            file_data,
            filename,
            folder_id=folder_id,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            progress_callback=progress_callback
        )
    except Exception as e:
        return False, f"Upload error: {str(e)}"

def upload_files_to_teams(files_to_upload, client_id, client_secret, tenant_id, finance_team_id,
                          access_token=None, token_expires_at=0, progress_callback=None):
    """Upload multiple files to the correct folders in Microsoft Teams based on filename pattern"""
    upload_results = []
    
    # First ensure the token is valid
    token, expires_at, _, error = ensure_valid_token(
        client_id, client_secret, tenant_id, access_token, token_expires_at)
    
    if error:
        return None, error, token, expires_at
    
    # Upload each file to the appropriate folder
    total_files = len(files_to_upload)
    for i, file_info in enumerate(files_to_upload):
        if progress_callback:
            progress_callback(f"Processing file {i+1}/{total_files}: {file_info['generated_filename']}")
        
        # Get original filename
        original_filename = file_info['generated_filename']
        
        # Determine target folder based on original filename pattern
        target_folder_id, folder_path, folder_name = determine_target_folder(
            original_filename,
            finance_team_id,
            token
        )
        
        if progress_callback:
            progress_callback(f"Target folder: {folder_name} (ID: {target_folder_id})")
            progress_callback(f"Folder path: {folder_path}")
        
        # Upload file with the folder ID approach
        success, error = upload_file(
            token, 
            None,  # drive_id not needed as we get it in upload function
            target_folder_id, 
            file_info['pdf_data'],
            original_filename,
            finance_team_id=finance_team_id,
            folder_path=folder_path,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            progress_callback=lambda msg: progress_callback(f"File {i+1}/{total_files}: {msg}")
            if progress_callback else None
        )
        
        upload_results.append({
            'filename': original_filename,
            'original_filename': file_info.get('original_filename', 'Unknown'),
            'success': success,
            'error': error,
            'target_folder': folder_name,
            'folder_id': target_folder_id
        })
    
    return upload_results, None, token, expires_at

def authenticate_teams(client_id, client_secret, tenant_id):
    """Authenticate with Microsoft Teams"""
    try:
        token, expires_at, app, error = ensure_valid_token(client_id, client_secret, tenant_id)
        if error:
            return None, 0, None, error
        
        return token, expires_at, app, None
    except Exception as e:
        return None, 0, None, f"Authentication error: {str(e)}"

def get_finance_team_folders(access_token, finance_team_id):
    """Get the specific folders within the Finance Team - kept for backward compatibility"""
    try:
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Get the team's SharePoint site
        response = requests.get(
            f'https://graph.microsoft.com/v1.0/groups/{finance_team_id}/sites/root',
            headers=headers
        )
        
        if response.status_code != 200:
            return None, None, None, f"Error getting Finance Team site: {response.status_code}"
        
        site_id = response.json()['id']
        
        # Get the drives in the site
        response = requests.get(
            f'https://graph.microsoft.com/v1.0/sites/{site_id}/drives',
            headers=headers
        )
        
        if response.status_code != 200:
            return None, None, None, f"Error getting site drives: {response.status_code}"
        
        # Get the documents drive (usually the first one)
        drives = response.json()['value']
        if not drives:
            return None, None, None, "No drives found in the Finance Team site"
        
        drive_id = drives[0]['id']
        
        # Use our known folder IDs instead of searching
        wmc_folder_id = "01OU6MNL3KE3XP2T5JMZC244U33CGKOAMH"  # WMC E-cheque
        client_trust_folder_id = "01OU6MNL5F3ZCFDTEACRAJDOX2G4NUMX72"  # E cheque WMC Nominee
        
        return drive_id, wmc_folder_id, client_trust_folder_id, None
    except Exception as e:
        return None, None, None, f"Failed to get Finance Team folders: {str(e)}"

def get_folder_contents(access_token, drive_id, folder_id):
    """Get contents of a specific folder"""
    try:
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f'https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children',
            headers=headers
        )
        
        if response.status_code != 200:
            return None, f"Error getting folder contents: {response.status_code}"
        
        items = response.json()['value']
        return items, None
    except Exception as e:
        return None, f"Failed to get folder contents: {str(e)}"
