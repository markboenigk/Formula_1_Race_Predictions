"""
Pipeline Orchestrator Lambda Entry Point.
Runs every 2 hours via EventBridge to:
1. Check for new session data (ingest if available)
2. Determine race context from DynamoDB
3. Check if predictions should run (qualifying complete, race pending)
4. Run inference if conditions met
"""
import logging
import os
import sys

# Add common to path for shared imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))

# Import orchestrator logic from local file
from orchestrator import lambda_handler as orchestrator_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    AWS Lambda entry point for the orchestrator.
    Wraps the core orchestrator logic.
    """
    logger.info(f"Orchestrator Lambda invoked with event: {event}")
    return orchestrator_handler(event, context)
