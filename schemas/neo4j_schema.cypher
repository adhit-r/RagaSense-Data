// Neo4j Schema for RagaSense-Data
// ================================
// This schema defines the structure for the RagaSense graph database
// All data will be populated from actual processed datasets

// Create constraints and indexes
CREATE CONSTRAINT raga_id_unique IF NOT EXISTS FOR (r:Raga) REQUIRE r.raga_id IS UNIQUE;
CREATE CONSTRAINT artist_id_unique IF NOT EXISTS FOR (a:Artist) REQUIRE a.artist_id IS UNIQUE;
CREATE CONSTRAINT composer_id_unique IF NOT EXISTS FOR (c:Composer) REQUIRE c.composer_id IS UNIQUE;
CREATE CONSTRAINT song_id_unique IF NOT EXISTS FOR (s:Song) REQUIRE s.song_id IS UNIQUE;
CREATE CONSTRAINT composition_id_unique IF NOT EXISTS FOR (rc:RagamalikaComposition) REQUIRE rc.composition_id IS UNIQUE;

// Create indexes for performance
CREATE INDEX raga_name_index IF NOT EXISTS FOR (r:Raga) ON (r.name);
CREATE INDEX raga_tradition_index IF NOT EXISTS FOR (r:Raga) ON (r.tradition);
CREATE INDEX artist_name_index IF NOT EXISTS FOR (a:Artist) ON (a.name);
CREATE INDEX composer_name_index IF NOT EXISTS FOR (c:Composer) ON (c.name);
CREATE INDEX song_title_index IF NOT EXISTS FOR (s:Song) ON (s.title);

// Node Labels and Properties:
// Raga: {raga_id, name, tradition, melakarta_number, parent_scale, arohana, avarohana, swaras, song_count, sources, metadata}
// Artist: {artist_id, name, tradition, song_count, specializations, period, sources, metadata}
// Composer: {composer_id, name, tradition, song_count, period, compositions, sources, metadata}
// Song: {song_id, title, tradition, duration, language, composition_form, sources, metadata}
// RagamalikaComposition: {composition_id, name, composer, type, tradition, total_ragas, unique_raga_count, constituent_ragas, metadata}
// CrossTraditionMapping: {mapping_id, relationship_type, similarity_score, notes, metadata}

// Relationship Types:
// SIMILAR_TO: Between Ragas (similarity_score, relationship_type)
// PERFORMS: Between Artist and Song (performance_date, venue)
// COMPOSED: Between Composer and Song (composition_date)
// IN_RAGA: Between Song and Raga (segment, duration)
// CONTAINS: Between RagamalikaComposition and Raga (segment_order, duration)
// MAPS: Between CrossTraditionMapping and Ragas

// Sample queries for testing (will work after data population):
// Find all ragas similar to a given raga
// MATCH (r1:Raga {name: $raga_name})-[:SIMILAR_TO]->(r2:Raga)
// RETURN r2.name, r2.tradition, r2.song_count;

// Find ragamalika compositions containing a specific raga
// MATCH (rc:RagamalikaComposition)-[:CONTAINS]->(r:Raga {name: $raga_name})
// RETURN rc.name, rc.composer, rc.type, rc.total_ragas;

// Find artists who perform in multiple ragas
// MATCH (a:Artist)-[:PERFORMS]->(s:Song)-[:IN_RAGA]->(r:Raga)
// WITH a, collect(DISTINCT r.name) as ragas
// WHERE size(ragas) > 5
// RETURN a.name, ragas, size(ragas) as raga_count;

// Find cross-tradition mappings
// MATCH (ctm:CrossTraditionMapping)-[:MAPS]->(r:Raga)
// RETURN ctm.relationship_type, ctm.similarity_score, r.name, r.tradition;

// Find ragamalika compositions with most ragas
// MATCH (rc:RagamalikaComposition)
// RETURN rc.name, rc.composer, rc.total_ragas, rc.unique_raga_count
// ORDER BY rc.total_ragas DESC;