import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from src.utils.config import CONFIG

def send_email_alert(subject, body):
    """Send an Email alert using SMTP."""
    sender_email = CONFIG['alerts']['sender_email']
    receiver_email = CONFIG['alerts']['receiver_email']
    password = CONFIG['alerts']['sender_password']
    smtp_server = CONFIG['alerts']['smtp_server']
    smtp_port = CONFIG['alerts']['smtp_port']
    
    if not sender_email or not password or str(sender_email).startswith("${"):
         print("Warning: Email credentials not configured. Email not sent.")
         print(f"Would have emailed:\nSubject: {subject}\nBody: {body}")
         return False
         
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"Email Alert sent to {receiver_email}.")
        return True
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        return False
