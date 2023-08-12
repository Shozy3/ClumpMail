import os.path
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import warnings
warnings.filterwarnings("ignore")


# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

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

        print(f"From: {sender}\nSubject: {subject}\nBody: {body[:100]}...\n{'-'*50}")

if __name__ == '__main__':
    main()

# This is the code for the PEGASUS NLP summarizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
text = """
Nomination Deadline is Fast Approaching on August 15th - Submit Today!
It only takes 20 minutes...
 

Since 1989, The ASTech Awards (ASTech.ca) has showcased Alberta’s excellence in science, technology and innovation by recognizing more than 600 individuals, organizations and teams at their annual prestigious awards event, bringing together industry, academia, government, and entrepreneurs to celebrate and reward excellence to inspire innovation.

Past Award Winners Include: SMART Technologies, AuroraWatch, Orpyx Medical Technologies, and many more.

Nominate a deserving individual, team, or company that inspires innovation in our province before it’s too late! 

Why Nominate a Deserving Award Candidate:
Help Grow Alberta's Tech Sector and Entrepreneurial Companies
Companies, Researchers, Regions, and Post-Secondaries Gain valuable peer and sector recognition
Greater Investment, Customers, Employees are attracted to nominated Innovators
Greater visibility to a wider pan-provincial cross-sector audience - Across Industry, Government, Academia
Credibility enhancement of all Alberta's Innovation Community 
Receive ongoing recognition as a leader in Alberta Science and Technology as an ASTech Alumni
Together, let’s inspire innovation through shared success and the best of all worlds.
"""
tokens = tokenizer(text, truncation = True, padding = "longest", return_tensors = "pt")
summary = model.generate(**tokens) 
print(tokenizer.decode(summary[0]))