"""
Script to recreate the DynamoDB tracking table with the correct schema.

WARNING: This will DELETE all existing data in the table!
Use this script only if you want to start fresh with the standardized schema.

Usage:
    python recreate_table.py [--table-name TABLE_NAME] [--backup-first]
"""

import boto3
import argparse
import json
import os
from datetime import datetime
from typing import Optional

# DynamoDB Table Configuration
DEFAULT_TABLE_NAME = 'f1_session_tracking'


def get_table_name() -> str:
    """Get the DynamoDB table name from environment variable or use default."""
    return os.environ.get('DYNAMODB_TABLE_NAME', DEFAULT_TABLE_NAME)

# Table schema definition
TABLE_SCHEMA = {
    'TableName': DEFAULT_TABLE_NAME,
    'KeySchema': [
        {
            'AttributeName': 'event_partition_key',
            'KeyType': 'HASH'  # Partition key
        },
        {
            'AttributeName': 'session_name_abr',
            'KeyType': 'RANGE'  # Sort key
        }
    ],
    'AttributeDefinitions': [
        {
            'AttributeName': 'event_partition_key',
            'AttributeType': 'S'  # String
        },
        {
            'AttributeName': 'session_name_abr',
            'AttributeType': 'S'  # String
        }
    ],
    'BillingMode': 'PAY_PER_REQUEST'  # On-demand billing
}


def backup_table(table_name: str, output_file: Optional[str] = None) -> int:
    """
    Backup all items from a DynamoDB table to a JSON file.
    
    Args:
        table_name: Name of the table to backup
        output_file: Optional output file path (defaults to timestamped filename)
    
    Returns:
        Number of items backed up
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'{table_name}_backup_{timestamp}.json'
    
    items = []
    scan_kwargs = {}
    
    while True:
        response = table.scan(**scan_kwargs)
        items.extend(response.get('Items', []))
        
        if 'LastEvaluatedKey' not in response:
            break
        
        scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
    
    with open(output_file, 'w') as f:
        json.dump(items, f, indent=2, default=str)
    
    print(f"Backed up {len(items)} items to {output_file}")
    return len(items)


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
        
        print(f"Table {table_name} deleted successfully")
    except dynamodb.exceptions.ResourceNotFoundException:
        print(f"Table {table_name} does not exist, skipping deletion")
    except Exception as e:
        print(f"Error deleting table: {str(e)}")
        raise


def create_table(table_name: str) -> None:
    """
    Create a new DynamoDB table with the correct schema.
    
    Args:
        table_name: Name of the table to create
    """
    dynamodb = boto3.client('dynamodb')
    
    schema = TABLE_SCHEMA.copy()
    schema['TableName'] = table_name
    
    try:
        print(f"Creating table {table_name}...")
        response = dynamodb.create_table(**schema)
        
        # Wait for table to be active
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        
        print(f"Table {table_name} created successfully")
        print(f"Table ARN: {response['TableDescription']['TableArn']}")
        print(f"Table Status: {response['TableDescription']['TableStatus']}")
    except dynamodb.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists")
    except Exception as e:
        print(f"Error creating table: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Recreate DynamoDB tracking table with standardized schema'
    )
    parser.add_argument(
        '--table-name',
        type=str,
        default=None,
        help=f'Table name (defaults to {DEFAULT_TABLE_NAME} or DYNAMODB_TABLE_NAME env var)'
    )
    parser.add_argument(
        '--backup-first',
        action='store_true',
        help='Backup existing table before deletion'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    table_name = args.table_name or get_table_name()
    
    print(f"Table name: {table_name}")
    print("\nWARNING: This will DELETE all existing data in the table!")
    
    if not args.force:
        confirmation = input("Are you sure you want to continue? (yes/no): ")
        if confirmation.lower() != 'yes':
            print("Aborted.")
            return
    
    try:
        # Backup if requested
        if args.backup_first:
            print("\nBacking up existing table...")
            backup_table(table_name)
        
        # Delete existing table
        print("\nDeleting existing table...")
        delete_table(table_name)
        
        # Create new table
        print("\nCreating new table...")
        create_table(table_name)
        
        print("\n✅ Table recreation complete!")
        print("\nNext steps:")
        print("1. Run create_tracking_table lambda to populate the table")
        print("2. Verify the table structure matches the schema")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
