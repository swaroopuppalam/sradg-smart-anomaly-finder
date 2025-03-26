import smtplib
import ssl
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

CONFIG_PATH = "/ml/email_config.json"

def load_email_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def send_email(subject, body, recipient):
    try:
        config = load_email_config()
        sender_email = config["sender_email"]
        receiver_email = recipient
        password = config["app_password"]
        smtp_server = config["smtp_server"]
        smtp_port = config["smtp_port"]

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        part = MIMEText(body, "html")
        msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
