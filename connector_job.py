"""
NairaMeter Automated Connector Job
====================================
Checks a dedicated email inbox for new partner data exports, processes any
found automatically through the existing, already-tested pipeline
(preprocess_steamaco_export + ingestion_service), and sends a summary email
when it's done — including immediately, loudly, on any failure.

This replaces manually running PowerShell commands each time a partner sends
new data. It changes WHO triggers the pipeline (a schedule, not a person) —
the actual processing logic is untouched and reuses the same tested code.

Designed to run periodically as a Render Cron Job (e.g., once per hour), not
as a permanently-running web service.

Environment variables required:
    IMAP_HOST, IMAP_USER, IMAP_PASSWORD   -> the inbox to check (e.g. a
                                              dedicated data@nairameter.com)
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD  -> same as backend_api_v4.py
    ALERT_EMAIL_TO                         -> where to send success/failure summaries
    SMTP_HOST, SMTP_USER, SMTP_PASSWORD    -> for sending that summary email

PARTNER_SENDER_MAP below is how the job knows which partner an email belongs
to — matched against the sender's address. Add a new partner by adding one
line here; no other code changes needed.
"""

import os
import re
import glob
import shutil
import tempfile
import zipfile
import imaplib
import email
import smtplib
from email.message import EmailMessage
from datetime import datetime

import pandas as pd

# Reuses the exact, already-tested merge logic — not reimplemented here.
from preprocess_steamaco_export import process_meter_folder
from ingestion_service import run_ingestion

PARTNER_SENDER_MAP = {
    # sender email address (lowercase) -> partner name used in the database
    "data@cesel.example.com": "cesel",
    # "data@huskpowersystems.com": "husk_power",   # add future partners here
}

def is_already_processed(message_id: str, conn) -> bool:
    """
    Tracks handled emails in the database itself, keyed by the email's own
    Message-ID header — not by the mailbox's read/unread status. This matters
    especially if the inbox being watched is also read by a real person (e.g.
    reusing info@nairameter.com rather than a dedicated address nobody
    checks): a human previewing an email in a normal mail client marks it
    read, which would make read-status-based tracking silently skip it
    forever. A database record can't be changed by accident that way.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM connector_processed_emails WHERE message_id = %s", (message_id,))
        return cur.fetchone() is not None


def mark_as_processed(message_id: str, conn, status: str, detail: str = ""):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO connector_processed_emails (message_id, processed_at, status, detail)
            VALUES (%s, now(), %s, %s)
            ON CONFLICT (message_id) DO UPDATE SET processed_at = now(), status = EXCLUDED.status, detail = EXCLUDED.detail;
        """, (message_id, status, detail))
    conn.commit()


def find_meter_folders(root_dir: str) -> dict:
    """
    Scans a directory (which may contain subfolders, or files directly) for
    SteamaCo-format files, grouping them by meter ID. Returns
    {meter_id: folder_path_containing_that_meters_files}.
    Handles both 'one zip per meter' and 'one zip with all meters inside'
    layouts, since we don't yet know which shape a given partner will send.
    """
    meter_dirs = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            match = re.match(r"Meter - (.+?) - .+\.csv", fname)
            if match:
                meter_id = match.group(1)
                meter_dirs[meter_id] = dirpath  # last folder containing this meter's files
    return meter_dirs


def process_attachment(zip_path: str, partner: str, cluster_id: str, customer_type: str, db_url: str) -> dict:
    """
    Extracts one email attachment, finds every meter's files inside it
    (however they're organised), merges each meter via the existing tested
    logic, combines them all, and runs the existing tested ingestion.
    Returns a small summary dict for the notification email.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        meter_dirs = find_meter_folders(tmpdir)
        if not meter_dirs:
            raise ValueError(f"No recognisable meter files found inside {os.path.basename(zip_path)}")

        combined_frames = []
        for meter_id, folder in meter_dirs.items():
            df = process_meter_folder(folder, cluster_id=cluster_id, customer_type=customer_type)
            combined_frames.append(df)

        combined = pd.concat(combined_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["customer_id", "date"])

        combined_csv = os.path.join(tmpdir, "combined_for_ingestion.csv")
        combined.to_csv(combined_csv, index=False)

        summary = run_ingestion(combined_csv, partner, db_url)
        summary["meters_found"] = list(meter_dirs.keys())
        return summary


def send_notification(subject: str, body: str):
    """Sends a plain summary email — success or failure — so a problem is
    never discovered by silence weeks later."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("ALERT_EMAIL_TO")
    if not all([smtp_host, smtp_user, smtp_password, to_addr]):
        print("SMTP not fully configured — skipping email notification. Summary:")
        print(body)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP_SSL(smtp_host, 465) as server:
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def run_connector_job():
    imap_host = os.environ["IMAP_HOST"]
    imap_user = os.environ["IMAP_USER"]
    imap_password = os.environ["IMAP_PASSWORD"]
    db_url = os.environ["DATABASE_URL"]

    import psycopg2
    conn = psycopg2.connect(db_url)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS connector_processed_emails (
                message_id TEXT PRIMARY KEY,
                processed_at TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL,
                detail TEXT
            );
        """)
    conn.commit()

    results = []
    errors = []

    mail = imaplib.IMAP4_SSL(imap_host)
    mail.login(imap_user, imap_password)
    mail.select("inbox")

    # Search ALL messages, not just unread ones — tracking is by database
    # record now (see is_already_processed), not by the mailbox's own
    # read/unread state. This is what makes it safe to share this inbox
    # with a real person's everyday correspondence: nothing about a human
    # opening or reading an email in their normal mail client affects
    # whether the job considers it "already handled".
    status, message_numbers = mail.search(None, "ALL")
    if status != "OK":
        raise RuntimeError("Could not search inbox")

    for num in message_numbers[0].split():
        status, msg_data = mail.fetch(num, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        message_id = msg.get("Message-ID", "").strip()
        if not message_id:
            continue  # can't safely track an email with no stable identifier

        if is_already_processed(message_id, conn):
            continue

        sender = email.utils.parseaddr(msg.get("From"))[1].lower()
        partner = PARTNER_SENDER_MAP.get(sender)

        if not partner:
            continue  # not from a recognised partner — leave it for a human, don't touch it at all

        found_zip = False
        for part in msg.walk():
            filename = part.get_filename()
            if filename and filename.lower().endswith(".zip"):
                found_zip = True
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    tmp_zip.write(part.get_payload(decode=True))
                    tmp_zip_path = tmp_zip.name
                try:
                    summary = process_attachment(
                        tmp_zip_path, partner=partner,
                        cluster_id=f"CL-{partner.upper()}-01", customer_type="residential",
                        db_url=db_url,
                    )
                    summary["source_email"] = sender
                    summary["filename"] = filename
                    results.append(summary)
                    mark_as_processed(message_id, conn, status="success", detail=f"{summary['rows_loaded']} rows loaded")
                except Exception as e:
                    errors.append(f"Failed processing '{filename}' from {sender}: {e}")
                    mark_as_processed(message_id, conn, status="failed", detail=str(e))
                    # Marked as processed even on failure — deliberately does NOT
                    # retry forever. A human gets the failure notification below
                    # and can re-send or investigate; endless silent retries of
                    # a genuinely broken file would just repeat the same failure.
                finally:
                    os.unlink(tmp_zip_path)

        if not found_zip:
            mark_as_processed(message_id, conn, status="skipped", detail="recognised sender, no zip attachment found")

    mail.logout()
    conn.close()

    # Always send a summary, even when nothing happened — silence should
    # never be the only signal that a scheduled job is still working.
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if errors:
        subject = f"[NairaMeter Connector] {len(errors)} failure(s) — {timestamp}"
    elif results:
        subject = f"[NairaMeter Connector] {len(results)} file(s) processed successfully — {timestamp}"
    else:
        subject = f"[NairaMeter Connector] No new data found — {timestamp}"

    body_lines = [f"Connector run at {timestamp}", ""]
    for r in results:
        body_lines.append(
            f"SUCCESS: {r['filename']} from {r['source_email']} — "
            f"partner={r['source_partner']}, meters={r.get('meters_found')}, "
            f"rows_loaded={r['rows_loaded']}/{r['rows_received']}, status={r['status']}"
        )
    for e in errors:
        body_lines.append(f"FAILURE: {e}")
    if not results and not errors:
        body_lines.append("No unread emails with recognised attachments since last run.")

    send_notification(subject, "\n".join(body_lines))
    return {"processed": len(results), "failed": len(errors)}


if __name__ == "__main__":
    outcome = run_connector_job()
    print(outcome)
