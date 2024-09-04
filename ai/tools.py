from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imaplib
import email
from email.parser import BytesParser
import smtplib
from typing import List
from langchain.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import Field
import json

from langchain_community.agent_toolkits.gmail.toolkit import GmailToolkit


class EmailReaderTool(BaseTool):
    name: str = "email_reader"
    description: str = """
    Use this tool to read the most recent emails from a specified email account.
    Input should be a JSON string with the following key:
    - n: (optional) number of recent emails to fetch (default is 5)
    
    The tool will return the contents of the n most recent emails, including subject, sender, date, and body.
    """
    username: str = Field(..., description="Email address")
    password: str = Field(..., description="Email password")
    server: str = Field(..., description="IMAP server address")

    def _run(self, input_str: str) -> str:
        try:
            params = json.loads(input_str)
            n = params.get('n', 5)
        except json.JSONDecodeError:
            return "Invalid input. Please provide a valid JSON string."

        try:
            # Connect to the email server
            mail = imaplib.IMAP4_SSL(self.server)
            mail.login(self.username, self.password)
            mail.select('inbox')

            # Search for the n most recent emails
            _, search_data = mail.search(None, 'ALL')
            msg_ids = search_data[0].split()
            msg_ids = msg_ids[-n:]  # Get the n most recent email ids

            results = []
            # Fetch and format the contents of the emails
            for msg_id in msg_ids:
                _, msg_data = mail.fetch(msg_id, '(RFC822)')
                raw_email = msg_data[0][1]
                email_message = BytesParser().parsebytes(raw_email)

                email_content = f"Subject: {email_message['Subject']}\n"
                email_content += f"From: {email_message['From']}\n"
                email_content += f"Date: {email_message['Date']}\n\n"

                for part in email_message.walk():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode('utf-8')
                        email_content += body

                results.append(email_content)

            mail.close()
            mail.logout()

            return "\n\n------------------------\n\n".join(results)

        except Exception as e:
            return f"An error occurred: {str(e)}"


# Usage
email_reader = EmailReaderTool(
    username='your_email@example.com',
    password='your_password',
    server='your_imap_server'
)


class EmailSenderTool(BaseTool):
    name: str = "email_sender"
    description: str = """
    Use this tool to send an email.
    Input should be a JSON string with the following keys:
    - to: recipient's email address
    - subject: email subject
    - body: email body content
    
    The tool will send the email and return a confirmation message.
    """
    username: str = Field(..., description="Sender's email address")
    password: str = Field(..., description="Sender's email password")
    server: str = Field(..., description="SMTP server address")
    port: int = Field(587, description="SMTP server port")

    def _run(self, input_str: str) -> str:
        try:
            params = json.loads(input_str)
            to_email = params['to']
            subject = params['subject']
            body = params['body']
        except (json.JSONDecodeError, KeyError):
            return "Invalid input. Please provide a valid JSON string with 'to', 'subject', and 'body' keys."

        try:
            # Create the email message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Connect to the SMTP server and send the email
            with smtplib.SMTP(self.server, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return f"Email sent successfully to {to_email}"

        except Exception as e:
            return f"An error occurred while sending the email: {str(e)}"


# Usage
email_sender = EmailSenderTool(
    username='your_email@example.com',
    password='your_password',
    server='your_smtp_server',
    port=587  # Adjust if your SMTP server uses a different port
)


class EmailToolKit(BaseToolkit):
    username: str = Field(..., description="Sender's email address")
    password: str = Field(..., description="Sender's email password")
    server: str = Field(..., description="SMTP server address")
    smtp_port: int = Field(587, description="SMTP server port")

    def get_tools(self) -> List[BaseTool]:
        return [
            EmailReaderTool(
                username=self.username,
                password=self.password,
                server=self.server
            ),
            EmailSenderTool(
                username=self.username,
                password=self.password,
                server=self.server,
                port=self.port
            )
        ]
