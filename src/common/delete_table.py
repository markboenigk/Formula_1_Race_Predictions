"""
Script to delete the DynamoDB tracking table.

WARNING: This will DELETE all existing data in the table!

Usage:
    python delete_table.py [--table-name TABLE_NAME] [--force]
"""

import boto3
import argparse
import os
from typing import Optional

# DynamoDB Table Configuration
DEFAULT_TABLE_NAME = 'f1_session_tracking'


def get_table_name() -> str:
    """Get the DynamoDB table name from environment variable or use default."""
    return os.environ.get('DYNAMODB_TABLE_NAME', DEFAULT_TABLE_NAME)


def delete_table(table_name: str) -> None:
    """
    Delete a DynamoDB table.
    
    Args:
        table_name: Name of the table to delete
    """
    dynamodb = boto3.client('dynamodb')
    
    try:
        print(f"Deleting table {table_name}...")
        dynamodb.delete_table(TableName=table_name)
        
        # Wait for table to be deleted
        waiter = dynamodb.get_waiter('table_not_exists')
        waiter.wait(TableName=table_name)
        
        print(f"✅ Table {table_name} deleted successfully")
    except dynamodb.exceptions.ResourceNotFoundException:
        print(f"⚠️  Table {table_name} does not exist, nothing to delete")
    except Exception as e:
        print(f"❌ Error deleting table: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Delete DynamoDB tracking table'
    )
    parser.add_argument(
        '--table-name',
        type=str,
        default=None,
        help=f'Table name (defaults to {DEFAULT_TABLE_NAME} or DYNAMODB_TABLE_NAME env var)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    table_name = args.table_name or get_table_name()
    
    print(f"Table name: {table_name}")
    print("\n⚠️  WARNING: This will DELETE all existing data in the table!")
    
    if not args.force:
        confirmation = input("Are you sure you want to continue? (yes/no): ")
        if confirmation.lower() != 'yes':
            print("Aborted.")
            return
    
    try:
        delete_table(table_name)
        print("\n✅ Table deletion complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
