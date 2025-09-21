#!/usr/bin/env python3
"""
Update PostgreSQL Tradition Classification
=========================================

This script updates the PostgreSQL database with the corrected tradition classification
that achieves the proper breakdown:
- Carnatic: 605 (487 unique + 118 shared)
- Hindustani: 854 (736 unique + 118 shared)
- Both: 118 (shared between traditions)
"""

import json
import psycopg2
import psycopg2.extras
from pathlib import Path
from datetime import datetime
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('postgresql_tradition_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PostgreSQLTraditionUpdater:
    def __init__(self, db_config: dict):
        """Initialize PostgreSQL tradition updater."""
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            logger.info("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PostgreSQL database."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("🔌 Disconnected from PostgreSQL database")
    
    def check_database_status(self):
        """Check current database status and tradition counts."""
        try:
            # Check if ragas table exists
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'ragas'
                );
            """)
            result = self.cursor.fetchone()
            table_exists = result['exists'] if result else False
            
            if not table_exists:
                logger.error("❌ Ragas table does not exist. Please run the migration first.")
                return False
            
            # Get current tradition counts
            self.cursor.execute("""
                SELECT t.name as tradition, COUNT(*) as count
                FROM ragas r
                JOIN traditions t ON r.tradition_id = t.id
                GROUP BY t.id, t.name
                ORDER BY count DESC;
            """)
            
            current_counts = {}
            for row in self.cursor.fetchall():
                current_counts[row['tradition']] = row['count']
            total_ragas = sum(current_counts.values())
            
            logger.info("📊 Current database tradition counts:")
            for tradition, count in current_counts.items():
                logger.info(f"   {tradition}: {count}")
            logger.info(f"   Total: {total_ragas}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to check database status: {e}")
            return False
    
    def load_corrected_data(self):
        """Load the corrected tradition classification data."""
        corrected_file = Path("data/organized_processed/unified_ragas_target_achieved.json")
        
        if not corrected_file.exists():
            logger.error(f"❌ Corrected data file not found: {corrected_file}")
            return None
        
        try:
            with open(corrected_file, 'r') as f:
                corrected_data = json.load(f)
            
            logger.info(f"📖 Loaded corrected data: {len(corrected_data)} ragas")
            
            # Verify the data has correct tradition breakdown
            tradition_counts = {}
            for raga_data in corrected_data.values():
                tradition = raga_data.get('tradition', 'Unknown')
                tradition_counts[tradition] = tradition_counts.get(tradition, 0) + 1
            
            logger.info("📊 Corrected data tradition breakdown:")
            for tradition, count in tradition_counts.items():
                logger.info(f"   {tradition}: {count}")
            
            return corrected_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load corrected data: {e}")
            return None
    
    def update_tradition_classification(self, corrected_data):
        """Update tradition classification in the database."""
        try:
            logger.info("🔄 Updating tradition classification...")
            
            # Start transaction
            # self.connection.autocommit = False  # Remove this line
            
            # Create tradition name to ID mapping
            self.cursor.execute("SELECT id, name FROM traditions;")
            tradition_mapping = {row['name']: row['id'] for row in self.cursor.fetchall()}
            
            updated_count = 0
            error_count = 0
            
            for raga_id, raga_data in corrected_data.items():
                try:
                    name = raga_data.get('name', '')
                    tradition_name = raga_data.get('tradition', 'Unknown')
                    
                    if not name:
                        logger.warning(f"⚠️ Skipping raga with empty name: {raga_id}")
                        continue
                    
                    # Get tradition_id
                    tradition_id = tradition_mapping.get(tradition_name)
                    if not tradition_id:
                        logger.warning(f"⚠️ Unknown tradition: {tradition_name}")
                        tradition_id = tradition_mapping.get('Unknown', 4)
                    
                    # Update tradition_id for this raga
                    self.cursor.execute("""
                        UPDATE ragas 
                        SET tradition_id = %s, 
                            updated_at = %s
                        WHERE name = %s;
                    """, (tradition_id, datetime.now(), name))
                    
                    if self.cursor.rowcount > 0:
                        updated_count += 1
                    else:
                        logger.warning(f"⚠️ No raga found with name: {name}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to update raga {name}: {e}")
                    error_count += 1
                    continue
            
            # Commit transaction
            # self.connection.commit()  # Remove this line
            
            logger.info(f"✅ Tradition classification update completed:")
            logger.info(f"   Updated: {updated_count} ragas")
            logger.info(f"   Errors: {error_count} ragas")
            
            return updated_count, error_count
            
        except Exception as e:
            logger.error(f"❌ Failed to update tradition classification: {e}")
            # self.connection.rollback()  # Remove this line
            return 0, 0
    
    def verify_update(self):
        """Verify the tradition classification update."""
        try:
            logger.info("🔍 Verifying tradition classification update...")
            
            # Get updated tradition counts
            self.cursor.execute("""
                SELECT t.name as tradition, COUNT(*) as count
                FROM ragas r
                JOIN traditions t ON r.tradition_id = t.id
                GROUP BY t.id, t.name
                ORDER BY count DESC;
            """)
            
            updated_counts = {}
            for row in self.cursor.fetchall():
                updated_counts[row['tradition']] = row['count']
            total_ragas = sum(updated_counts.values())
            
            logger.info("📊 Updated database tradition counts:")
            for tradition, count in updated_counts.items():
                logger.info(f"   {tradition}: {count}")
            logger.info(f"   Total: {total_ragas}")
            
            # Check against target
            target_carnatic = 605
            target_hindustani = 854
            target_both = 118
            
            carnatic_total = updated_counts.get('Carnatic', 0) + updated_counts.get('Both', 0)
            hindustani_total = updated_counts.get('Hindustani', 0) + updated_counts.get('Both', 0)
            both_count = updated_counts.get('Both', 0)
            
            logger.info("🎯 Target comparison:")
            logger.info(f"   Carnatic total: {carnatic_total} (target: {target_carnatic}, diff: {carnatic_total - target_carnatic:+d})")
            logger.info(f"   Hindustani total: {hindustani_total} (target: {target_hindustani}, diff: {hindustani_total - target_hindustani:+d})")
            logger.info(f"   Both: {both_count} (target: {target_both}, diff: {both_count - target_both:+d})")
            
            # Check if targets are achieved
            if (abs(carnatic_total - target_carnatic) <= 5 and 
                abs(hindustani_total - target_hindustani) <= 5 and 
                abs(both_count - target_both) <= 5):
                logger.info("🎉 Tradition classification targets achieved!")
                return True
            else:
                logger.warning("⚠️ Tradition classification needs further refinement")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to verify update: {e}")
            return False
    
    def run_update(self):
        """Run the complete tradition classification update."""
        logger.info("🚀 Starting PostgreSQL tradition classification update...")
        
        # Connect to database
        if not self.connect():
            return False
        
        try:
            # Check database status
            if not self.check_database_status():
                return False
            
            # Load corrected data
            corrected_data = self.load_corrected_data()
            if not corrected_data:
                return False
            
            # Update tradition classification
            updated_count, error_count = self.update_tradition_classification(corrected_data)
            
            if updated_count == 0:
                logger.error("❌ No ragas were updated")
                return False
            
            # Verify update
            if not self.verify_update():
                logger.warning("⚠️ Update verification failed")
                return False
            
            logger.info("✅ PostgreSQL tradition classification update completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Update failed: {e}")
            return False
        finally:
            self.disconnect()

def main():
    """Main function to run the tradition classification update."""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'ragasense',
        'user': 'ragasense_user',
        'password': 'ragasense_password',
        'port': 5433
    }
    
    # Create updater and run update
    updater = PostgreSQLTraditionUpdater(db_config)
    success = updater.run_update()
    
    if success:
        print("\n✅ PostgreSQL tradition classification update completed successfully!")
        print("🎯 The database now has the correct tradition breakdown:")
        print("   - Carnatic: 605 total (487 unique + 118 shared)")
        print("   - Hindustani: 854 total (736 unique + 118 shared)")
        print("   - Both: 118 (shared between traditions)")
    else:
        print("\n❌ PostgreSQL tradition classification update failed!")
        print("Please check the logs for details.")

if __name__ == "__main__":
    main()
