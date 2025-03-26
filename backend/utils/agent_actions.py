import json
import os
import time

AGENT_LOG_PATH = "/shared/agent_log.json"
ALERT_RECIPIENT = "recon-team@example.com"

def create_ticket(anomaly_id, reason):
    task = {
        "action": "Create JIRA Ticket",
        "reason": reason,
        "anomaly_id": anomaly_id,
        "status": "submitted",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_action(task)

def send_email(anomaly_id, reason):
    task = {
        "action": "Send Email Alert",
        "reason": reason,
        "anomaly_id": anomaly_id,
        "status": "sent",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_action(task)

def send_email_alert(anomaly_id, reason):
    subject = f"🚨 Anomaly Detected [{reason}]"
    body = f"""
    <h3>Anomaly Triggered</h3>
    <p><b>ID:</b> {anomaly_id}</p>
    <p><b>Reason:</b> {reason}</p>
    <p>Please take necessary action.</p>
    """
    success = send_email(subject, body, ALERT_RECIPIENT)
    task = {
        "action": "Send Email Alert",
        "reason": reason,
        "anomaly_id": anomaly_id,
        "status": "sent" if success else "failed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_action(task)

def create_resolution_task(anomaly_id, reason):
    task = {
        "action": "Create Resolution Task",
        "reason": reason,
        "anomaly_id": anomaly_id,
        "status": "queued",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_action(task)

def log_action(task):
    logs = []
    if os.path.exists(AGENT_LOG_PATH):
        try:
            with open(AGENT_LOG_PATH) as f:
                logs = json.load(f)
        except:
            pass
    logs.append(task)
    with open(AGENT_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)
