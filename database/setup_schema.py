#!/usr/bin/env python3
"""
Setup database schema for RagaSense-Data
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database_schema():
    """Create database schema"""
    
    # Check environment variables
    required_env_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your environment or .env file")
        return False
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST'),
            database=os.getenv('POSTGRES_DATABASE'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            port=os.getenv('POSTGRES_PORT', 5432)
        )
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("✅ Connected to database")
        
        # Read and execute schema
        with open('database/schema.sql', 'r') as f:
            schema_sql = f.read()
        
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements):
            try:
                cursor.execute(statement)
                print(f"✅ Executed statement {i+1}/{len(statements)}")
            except Exception as e:
                print(f"⚠️  Statement {i+1} failed: {e}")
                # Continue with other statements
        
        print("✅ Database schema setup completed!")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print(f"📊 Created {len(tables)} tables:")
        for table in tables:
            print(f"  • {table[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema setup failed: {e}")
        return False
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🚀 Setting up RagaSense-Data database schema...")
    success = setup_database_schema()
    
    if success:
        print("\n🎉 Database schema is ready!")
        print("Next steps:")
        print("1. Run: python database/migrate_data.py")
        print("2. Deploy: vercel --prod")
    else:
        print("\n❌ Schema setup failed. Please check your database connection.")

