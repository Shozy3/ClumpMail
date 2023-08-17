from __future__ import print_function

import os
import base64
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.message import EmailMessage

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers', message=".*weights of PegasusForConditionalGeneration were not initialized.*")


import logging
logging.basicConfig(level=logging.ERROR)



# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']

script_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer_path = os.path.join(script_dir, "local_pegasus_tokenizer")
model_path = os.path.join(script_dir, "local_pegasus_model")

tokenizer = PegasusTokenizer.from_pretrained(tokenizer_path)
model = PegasusForConditionalGeneration.from_pretrained(model_path)


def get_service():
    """
    Verify that the user has the neccesary credentials for interacing with the Gmail API
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service


def main():
    service = get_service()
    results = service.users().messages().list(userId='me', labelIds=['CATEGORY_PERSONAL'], maxResults=5).execute()
    messages = results.get('messages', [])
    return_email = ""

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()

        for header in msg['payload']['headers']:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']
        try:
            part = [part for part in msg['payload']['parts'] if part['mimeType'] == 'text/plain'][0]
            body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        except:
            continue
            body = "Could not decode body."

        tokens = tokenizer(body, truncation=True, padding="longest", return_tensors="pt")
        summary_encoded = model.generate(**tokens)
        summary_decoded = tokenizer.decode(summary_encoded[0])

        return_email += "Subject: " + subject + '\n\n'
        return_email += "Summary: " + summary_decoded + '\n\n\n\n'

    message = EmailMessage()
    message.set_content(return_email)
    message['To'] = 'shahzaib@ualberta.ca'
    message['From'] = 'shahzaib@ualberta.ca'
    message['Subject'] = 'Test'
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    message_dict = {"raw": encoded_message}
    try:
        message = (service.users().messages().send(userId='me', body=message_dict).execute())
        print('Message Sent!')
    except HttpError as error:
        print("An error occured: %s" % error)



if __name__ == '__main__':
    main()