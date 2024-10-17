import imaplib
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.parser import BytesParser

from langchain.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import Field


class EmailReaderTool(BaseTool):
    name: str = "email_reader"
    description: str = """
    Use this tool to read the most recent emails from the users inbox.
    Input should be the number of recent emails to fetch (default is 5)
    
    The tool will return the contents of the n most recent emails, including subject, sender, date, and body.
    """
    username: str = Field(..., description="Email address")
    password: str = Field(..., description="Email password")
    server: str = Field(..., description="IMAP server address")

    def _run(self, n: int = 5) -> str:
        print(f"\n--- EmailReaderTool: _run started with n={n} ---")
        try:
            print(f"Connecting to email server: {self.server}...")
            # Connect to the email server
            mail = imaplib.IMAP4_SSL(self.server)
            print(f"Logging in with username: {self.username}...")
            mail.login(self.username, self.password)
            print(f"Selecting inbox...")
            mail.select("inbox")

            print(f"Searching for {n} most recent emails...")
            # Search for the n most recent emails
            _, search_data = mail.search(None, "ALL")
            msg_ids = search_data[0].split()
            msg_ids = msg_ids[-n:]  # Get the n most recent email ids
            print(f"Found {len(msg_ids)} email ids.")

            results = []
            print(f"Fetching and formatting email contents...")
            # Fetch and format the contents of the emails
            for i, msg_id in enumerate(msg_ids):
                print(f"Fetching email {i+1} of {n}...")
                _, msg_data = mail.fetch(msg_id, "(RFC822)")
                raw_email = msg_data[0][1]
                email_message = BytesParser().parsebytes(raw_email)

                email_content = f"Subject: {email_message['Subject']}\n"
                email_content += f"From: {email_message['From']}\n"
                email_content += f"Date: {email_message['Date']}\n\n"

                print(f"Extracting email body...")
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        charset = part.get_content_charset()
                        if charset is None:
                            charset = "utf-8"
                        body = part.get_payload(decode=True).decode(charset, errors="replace")
                        email_content += body

                results.append(email_content)
                print(f"Email {i+1} of {n} processed.")

            print(f"Closing email connection...")
            mail.close()
            mail.logout()

            print(f"Returning email contents...")
            return "\n\n------------------------\n\n".join(results)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"

        finally:
            print(f"--- EmailReaderTool: _run completed ---")


class EmailSenderTool(BaseTool):
    name: str = "email_sender"
    description: str = """
    Send an email to a specified recipient.

    Args:
        to_email (str): The recipient's email address.
        subject (str): The subject of the email.
        body (str): The body content of the email.

    Returns:
        str: A confirmation message indicating whether the email was sent successfully or if an error occurred.

    Raises:
        Exception: If there's an error in sending the email, the exception message will be returned.
    """
    username: str = Field(..., description="Sender's email address")
    password: str = Field(..., description="Sender's email password")
    server: str = Field(..., description="SMTP server address")
    port: int = Field(587, description="SMTP server port")

    def _run(self, to_email: str, subject: str, body: str) -> str:
        print(f"EmailSenderTool received input - To: {to_email}, Subject: {subject}")
        # Print first 50 characters of body
        print(f"Email body: {body[:50]}...")

        try:
            print("Creating email message...")
            msg = MIMEMultipart()
            msg["From"] = self.username
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            print("Email message created successfully.")

            print(f"Connecting to SMTP server: {self.server}:{self.port}")
            with smtplib.SMTP(self.server, self.port, local_hostname="client1.avangenio.com") as server:
                print("Starting TLS...")
                server.starttls()
                print(f"Logging in as {self.username}")
                server.login(self.username, self.password)
                print("Sending message...")
                server.send_message(msg)
                print("Message sent successfully.")

            return f"Email sent successfully to {to_email}"

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"An error occurred while sending the email: {str(e)}"


class EmailToolkit(BaseToolkit):
    """
    A toolkit for email-related operations, providing tools for reading and sending emails.

    This toolkit includes two tools:
    1. EmailReaderTool: For reading recent emails from the user's inbox.
    2. EmailSenderTool: For sending emails to specified recipients.

    Attributes:
        username (str): The email address used for both sending and reading emails.
        password (str): The password for the email account.
        server (str): The server address for both IMAP (reading) and SMTP (sending) operations.
        smtp_port (int): The port number for the SMTP server, defaulting to 587.

    Methods:
        get_tools(): Returns a list of the email tools (EmailReaderTool and EmailSenderTool).
    """

    username: str = Field(..., description="Email address for sending and reading emails")
    password: str = Field(..., description="Password for the email account")
    server: str = Field(..., description="Server address for IMAP and SMTP operations")
    smtp_port: int = Field(587, description="SMTP server port")

    def get_tools(self) -> list[BaseTool]:
        """
        Instantiates and returns a list of email-related tools.
        Returns:
            List[BaseTool]: A list containing instances of EmailReaderTool and EmailSenderTool.
        """
        return [
            EmailReaderTool(username=self.username, password=self.password, server=self.server),
            EmailSenderTool(
                username=self.username,
                password=self.password,
                server=self.server,
                port=self.smtp_port,
            ),
        ]
