# gmail_component.py
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64
import os
import tempfile
from datetime import datetime, timedelta

# Gmail API scopes - we only need readonly for searching and downloading
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service(gmail_secrets):
    """Initialize Gmail service with token-based authentication from secrets"""
    try:
        # Create credentials directly from tokens passed as parameters
        creds = Credentials(
            token=gmail_secrets["token"],
            refresh_token=gmail_secrets["refresh_token"],
            token_uri=gmail_secrets["token_uri"],
            client_id=gmail_secrets["client_id"],
            client_secret=gmail_secrets["client_secret"],
            scopes=SCOPES
        )
        
        # Build the service
        service = build('gmail', 'v1', credentials=creds)
        return service, None
            
    except Exception as e:
        return None, f"Gmail service initialization error: {str(e)}"

def search_echeque_emails(service, start_date, end_date):
    """Search for BOCHK e-Cheque emails within a date range."""
    # Fixed subject
    subject = "BOCHK e-Cheque"
    
    # Format dates for Gmail query
    start_date_str = start_date.strftime('%Y/%m/%d')
    end_date_str = end_date.strftime('%Y/%m/%d')
    
    # Create query
    query = f'subject:"{subject}" after:{start_date_str} before:{end_date_str}'
    
    try:
        # Execute search
        result = service.users().messages().list(userId='me', q=query).execute()
        messages = result.get('messages', [])
        
        # Get more messages if there are any
        while 'nextPageToken' in result:
            page_token = result['nextPageToken']
            result = service.users().messages().list(userId='me', q=query, pageToken=page_token).execute()
            messages.extend(result.get('messages', []))
        
        return messages, None
    except Exception as e:
        return None, f"Error searching emails: {str(e)}"

def get_email_details(service, msg_id):
    """Get details of a specific email."""
    try:
        message = service.users().messages().get(userId='me', id=msg_id).execute()
        
        # Extract headers
        headers = message['payload']['headers']
        subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
        sender = next((header['value'] for header in headers if header['name'] == 'From'), 'Unknown')
        date = next((header['value'] for header in headers if header['name'] == 'Date'), 'Unknown')
        
        return {
            'id': msg_id,
            'subject': subject,
            'sender': sender,
            'date': date,
            'message': message
        }, None
    except Exception as e:
        return None, f"Error getting email details: {str(e)}"

def download_attachments(service, message, download_dir):
    """Download attachments from a specific email."""
    try:
        attachments = []
        msg_id = message['id']
        
        # Get the full message data if not already included
        if 'message' in message and isinstance(message['message'], dict):
            message_data = message['message']
        else:
            message_data = service.users().messages().get(userId='me', id=msg_id).execute()
        
        # Check if there are any parts
        if 'parts' not in message_data['payload']:
            return [], None
        
        parts = message_data['payload']['parts']
        
        for part in parts:
            if part.get('filename') and part.get('body') and part['body'].get('attachmentId'):
                attachment_id = part['body']['attachmentId']
                filename = part['filename']
                
                # Get attachment
                attachment = service.users().messages().attachments().get(
                    userId='me', messageId=msg_id, id=attachment_id).execute()
                
                # Decode attachment data
                file_data = base64.urlsafe_b64decode(attachment['data'])
                
                # Save attachment
                filepath = os.path.join(download_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                
                attachments.append({
                    'filename': filename,
                    'path': filepath,
                    'size': len(file_data),
                    'content': file_data  # Store the content for later use
                })
        
        return attachments, None
    except Exception as e:
        return None, f"Error downloading attachments: {str(e)}"

def search_and_download_echeques(gmail_secrets, start_date, end_date, progress_callback=None):
    """Main function to search and download e-cheques from Gmail.
    
    Args:
        gmail_secrets: Dictionary containing Gmail API credentials
        start_date: Start date for email search
        end_date: End date for email search
        progress_callback: Optional function to call with progress updates
        
    Returns:
        (downloaded_files, error_message)
    """
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Initialize Gmail service
    service, error = get_gmail_service(gmail_secrets)
    if error:
        return None, error
    
    # Search for e-cheque emails
    if progress_callback:
        progress_callback("Searching for e-cheque emails...")
    
    messages, error = search_echeque_emails(service, start_date, end_date)
    if error:
        return None, error
    
    if not messages:
        return [], "No e-cheques found in the date range."
    
    # Download attachments
    downloaded_files = []
    
    total_messages = len(messages)
    for i, msg in enumerate(messages):
        if progress_callback:
            progress_callback(f"Processing email {i+1}/{total_messages}...")
        
        # Get email details
        email_details, error = get_email_details(service, msg['id'])
        if error:
            continue
        
        # Download attachments
        attachments, error = download_attachments(service, email_details, temp_dir)
        if error:
            continue
        
        # Add to downloaded files
        for attachment in attachments:
            downloaded_files.append({
                'email_subject': email_details['subject'],
                'email_date': email_details['date'],
                'email_sender': email_details['sender'],
                'filename': attachment['filename'],
                'path': attachment['path'],
                'size': attachment['size'],
                'content': attachment['content']
            })
    
    return downloaded_files, None
