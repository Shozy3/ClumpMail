import os
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers', message=".*weights of PegasusForConditionalGeneration were not initialized.*")

import logging
logging.basicConfig(level=logging.ERROR)



# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

script_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer_path = os.path.join(script_dir, "local_pegasus_tokenizer")
model_path = os.path.join(script_dir, "local_pegasus_model")

tokenizer = PegasusTokenizer.from_pretrained(tokenizer_path)
model = PegasusForConditionalGeneration.from_pretrained(model_path)


def get_service():
    creds = None

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')

    # If there are no (valid) credentials available, prompt the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service

def main():
    service = get_service()

    # Get the list of messages
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()

        # Decode the email subject, sender, and body
        for header in msg['payload']['headers']:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']

        # Assuming the message is simple and has a text/plain part in 'parts'
        try:
            part = [part for part in msg['payload']['parts'] if part['mimeType'] == 'text/plain'][0]
            body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        except:
            body = "Could not decode body."

        # print(f"From: {sender}\nSubject: {subject}\nBody: {body[:100]}...\n{'-'*50}")

        emails = {}
        emails[subject] = body

        tokens = tokenizer(body, truncation = True, padding = "longest", return_tensors = "pt")
        summary_encoded = model.generate(**tokens) 

        summary_decoded = tokenizer.decode(summary_encoded[0])

        summarized = {}
        summarized[subject] = summary_decoded
    
        print(subject)
        print(summary_decoded)

if __name__ == '__main__':
    main()
