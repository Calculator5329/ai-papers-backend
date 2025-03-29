import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_recent():
    logger.info("🔄 Updating recent papers...")
    try:
        result = subprocess.run(["python3", "fill_recent_db.py"], capture_output=True, text=True, check=True)
        logger.info(f"fill_recent_db.py output:\n{result.stdout}")  # Log stdout
        if result.stderr:
            logger.error(f"fill_recent_db.py error:\n{result.stderr}") # Log stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running fill_recent_db.py: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")

def update_daily():
    logger.info("🔄 Updating daily papers...")
    try:
        result = subprocess.run(["python3", "fill_daily_db.py"], capture_output=True, text=True, check=True)
        logger.info(f"fill_daily_db.py output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"fill_daily_db.py error:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running fill_daily_db.py: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")

def update_weekly():
    logger.info("🔄 Updating weekly papers...")
    try:
        result = subprocess.run(["python3", "fill_weekly_db.py"], capture_output=True, text=True, check=True)
        logger.info(f"fill_weekly_db.py output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"fill_weekly_db.py error:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running fill_weekly_db.py: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
def run_scheduled_tasks():
    logger.info("🚀 Running all scheduled tasks...")
    update_recent()
    update_daily()
    update_weekly()
    logger.info("✅ Successfully executed all scheduled tasks.")