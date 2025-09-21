#!/usr/bin/env python3
"""
Migrate RagaSense data to Vercel Postgres database
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
from datetime import datetime
import sys

def get_db_connection():
    """Get database connection from environment variables"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        database=os.getenv('POSTGRES_DATABASE'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        port=os.getenv('POSTGRES_PORT', 5432)
    )

def load_unified_dataset():
    """Load the latest unified dataset"""
    dataset_files = []
    for file in os.listdir('data/04_ml_datasets/unified/'):
        if file.startswith('unified_raga_dataset_') and file.endswith('.json'):
            dataset_files.append(file)
    
    if not dataset_files:
        raise FileNotFoundError("No unified dataset found")
    
    # Get the latest dataset
    latest_file = sorted(dataset_files)[-1]
    print(f"Loading dataset: {latest_file}")
    
    with open(f'data/04_ml_datasets/unified/{latest_file}', 'r') as f:
        return json.load(f)

def migrate_ragas(conn, dataset):
    """Migrate ragas data"""
    print("Migrating ragas...")
    
    cursor = conn.cursor()
    
    # Get all ragas from the dataset
    ragas_data = dataset.get('ragas', {})
    
    for raga_name, raga_info in ragas_data.items():
        try:
            # Determine tradition and source
            tradition = 'both'
            source = 'both'
            
            if 'tradition' in raga_info:
                tradition = raga_info['tradition'].lower()
            elif 'sources' in raga_info:
                sources = raga_info['sources']
                if 'ramanarunachalam' in sources and 'saraga' in sources:
                    tradition = 'both'
                    source = 'both'
                elif 'ramanarunachalam' in sources:
                    tradition = 'carnatic'  # Default for Ramanarunachalam
                    source = 'ramanarunachalam'
                elif 'saraga' in sources:
                    tradition = 'hindustani'  # Default for Saraga
                    source = 'saraga'
            
            # Extract additional info
            arohana = raga_info.get('arohana', '')
            avarohana = raga_info.get('avarohana', '')
            melakarta = raga_info.get('melakarta', '')
            janya_type = raga_info.get('janya_type', '')
            song_count = raga_info.get('song_count', 0)
            
            # Create metadata
            metadata = {
                'original_data': raga_info,
                'migrated_at': datetime.now().isoformat()
            }
            
            cursor.execute("""
                INSERT INTO ragas (name, tradition, source, arohana, avarohana, 
                                 melakarta, janya_type, song_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name, tradition, source) DO UPDATE SET
                    arohana = EXCLUDED.arohana,
                    avarohana = EXCLUDED.avarohana,
                    melakarta = EXCLUDED.melakarta,
                    janya_type = EXCLUDED.janya_type,
                    song_count = EXCLUDED.song_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (raga_name, tradition, source, arohana, avarohana, 
                  melakarta, janya_type, song_count, json.dumps(metadata)))
            
        except Exception as e:
            print(f"Error migrating raga {raga_name}: {e}")
            continue
    
    conn.commit()
    print(f"Migrated {len(ragas_data)} ragas")

def migrate_artists(conn, dataset):
    """Migrate artists data"""
    print("Migrating artists...")
    
    cursor = conn.cursor()
    
    # Get artists from the dataset
    artists_data = dataset.get('artists', {})
    
    for artist_name, artist_info in artists_data.items():
        try:
            tradition = artist_info.get('tradition', 'both').lower()
            source = artist_info.get('source', 'both').lower()
            song_count = artist_info.get('song_count', 0)
            
            metadata = {
                'original_data': artist_info,
                'migrated_at': datetime.now().isoformat()
            }
            
            cursor.execute("""
                INSERT INTO artists (name, tradition, source, song_count, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name, tradition, source) DO UPDATE SET
                    song_count = EXCLUDED.song_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (artist_name, tradition, source, song_count, json.dumps(metadata)))
            
        except Exception as e:
            print(f"Error migrating artist {artist_name}: {e}")
            continue
    
    conn.commit()
    print(f"Migrated {len(artists_data)} artists")

def migrate_songs(conn, dataset):
    """Migrate songs data"""
    print("Migrating songs...")
    
    cursor = conn.cursor()
    
    # Get songs from the dataset
    songs_data = dataset.get('songs', {})
    
    for song_title, song_info in songs_data.items():
        try:
            # Get raga and artist IDs
            raga_name = song_info.get('raga', '')
            artist_name = song_info.get('artist', '')
            
            # Find raga ID
            cursor.execute("SELECT id FROM ragas WHERE name = %s LIMIT 1", (raga_name,))
            raga_result = cursor.fetchone()
            raga_id = raga_result[0] if raga_result else None
            
            # Find artist ID
            cursor.execute("SELECT id FROM artists WHERE name = %s LIMIT 1", (artist_name,))
            artist_result = cursor.fetchone()
            artist_id = artist_result[0] if artist_result else None
            
            if not raga_id or not artist_id:
                print(f"Skipping song {song_title} - missing raga or artist")
                continue
            
            composer = song_info.get('composer', '')
            tradition = song_info.get('tradition', 'both').lower()
            source = song_info.get('source', 'both').lower()
            youtube_links = song_info.get('youtube_links', [])
            audio_features = song_info.get('audio_features', {})
            
            metadata = {
                'original_data': song_info,
                'migrated_at': datetime.now().isoformat()
            }
            
            cursor.execute("""
                INSERT INTO songs (title, raga_id, artist_id, composer, tradition, 
                                 source, youtube_links, audio_features, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (title, raga_id, artist_id) DO UPDATE SET
                    composer = EXCLUDED.composer,
                    tradition = EXCLUDED.tradition,
                    source = EXCLUDED.source,
                    youtube_links = EXCLUDED.youtube_links,
                    audio_features = EXCLUDED.audio_features,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (song_title, raga_id, artist_id, composer, tradition, 
                  source, youtube_links, json.dumps(audio_features), json.dumps(metadata)))
            
        except Exception as e:
            print(f"Error migrating song {song_title}: {e}")
            continue
    
    conn.commit()
    print(f"Migrated {len(songs_data)} songs")

def migrate_cross_tradition_mappings(conn, dataset):
    """Migrate existing cross-tradition mappings"""
    print("Migrating cross-tradition mappings...")
    
    cursor = conn.cursor()
    
    # Get cross-tradition mappings
    mappings = dataset.get('cross_tradition_mappings', {})
    
    # Process exact matches
    exact_matches = mappings.get('exact_matches', [])
    for raga_name in exact_matches:
        try:
            # Find both Saraga and Ramanarunachalam versions
            cursor.execute("""
                SELECT id FROM ragas 
                WHERE name = %s AND source IN ('saraga', 'both')
                LIMIT 1
            """, (raga_name,))
            saraga_result = cursor.fetchone()
            
            cursor.execute("""
                SELECT id FROM ragas 
                WHERE name = %s AND source IN ('ramanarunachalam', 'both')
                LIMIT 1
            """, (raga_name,))
            ramanarunachalam_result = cursor.fetchone()
            
            if saraga_result and ramanarunachalam_result:
                cursor.execute("""
                    INSERT INTO cross_tradition_mappings 
                    (saraga_raga_id, ramanarunachalam_raga_id, relationship_type, 
                     similarity_score, confidence, status)
                    VALUES (%s, %s, 'exact_match', 1.0, 1.0, 'approved')
                    ON CONFLICT DO NOTHING
                """, (saraga_result[0], ramanarunachalam_result[0]))
            
        except Exception as e:
            print(f"Error migrating exact match {raga_name}: {e}")
            continue
    
    # Process similar matches
    similar_matches = mappings.get('similar_matches', [])
    for match_pair in similar_matches:
        try:
            if len(match_pair) != 2:
                continue
                
            saraga_name, ramanarunachalam_name = match_pair
            
            cursor.execute("""
                SELECT id FROM ragas 
                WHERE name = %s AND source IN ('saraga', 'both')
                LIMIT 1
            """, (saraga_name,))
            saraga_result = cursor.fetchone()
            
            cursor.execute("""
                SELECT id FROM ragas 
                WHERE name = %s AND source IN ('ramanarunachalam', 'both')
                LIMIT 1
            """, (ramanarunachalam_name,))
            ramanarunachalam_result = cursor.fetchone()
            
            if saraga_result and ramanarunachalam_result:
                cursor.execute("""
                    INSERT INTO cross_tradition_mappings 
                    (saraga_raga_id, ramanarunachalam_raga_id, relationship_type, 
                     similarity_score, confidence, status)
                    VALUES (%s, %s, 'similar_match', 0.8, 0.8, 'approved')
                    ON CONFLICT DO NOTHING
                """, (saraga_result[0], ramanarunachalam_result[0]))
            
        except Exception as e:
            print(f"Error migrating similar match {match_pair}: {e}")
            continue
    
    conn.commit()
    print(f"Migrated {len(exact_matches)} exact matches and {len(similar_matches)} similar matches")

def main():
    """Main migration function"""
    print("🚀 Starting RagaSense data migration to Vercel Postgres...")
    
    # Check environment variables
    required_env_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your Vercel environment or .env file")
        sys.exit(1)
    
    try:
        # Load dataset
        dataset = load_unified_dataset()
        
        # Connect to database
        conn = get_db_connection()
        print("✅ Connected to database")
        
        # Run migrations
        migrate_ragas(conn, dataset)
        migrate_artists(conn, dataset)
        migrate_songs(conn, dataset)
        migrate_cross_tradition_mappings(conn, dataset)
        
        print("✅ Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()

