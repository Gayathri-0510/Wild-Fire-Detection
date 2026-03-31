from twilio.rest import Client
from src.utils.config import CONFIG

def send_sms_alert(message_body):
    """Send SMS alert using Twilio."""
    account_sid = CONFIG['alerts']['twilio_account_sid']
    auth_token = CONFIG['alerts']['twilio_auth_token']
    from_num = CONFIG['alerts']['twilio_from_number']
    to_num = CONFIG['alerts']['twilio_to_number']
    
    # Simple validation so script doesn't crash if config is missing
    if not account_sid or not auth_token or account_sid.startswith("${"):
        print("Warning: Twilio credentials not configured. SMS not sent.")
        print(f"Message that would have been sent: {message_body}")
        return False
        
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message_body,
            from_=from_num,
            to=to_num
        )
        print(f"SMS Alert sent to {to_num}. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"Failed to send SMS trigger: {e}")
        return False
