// RagaSense-Data: Neo4j Graph Schema for Raga Relationships
// This schema defines the graph structure for mapping relationships between ragas
// NO HARDCODED DATA - All data will be populated from actual processed datasets

// =============================================================================
// NODE LABELS AND PROPERTIES
// =============================================================================

// Raga nodes - represent individual ragas from both traditions
CREATE CONSTRAINT raga_id_unique IF NOT EXISTS FOR (r:Raga) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT raga_name_exists IF NOT EXISTS FOR (r:Raga) REQUIRE r.name IS NOT NULL;

// Raga node structure
// (:Raga {
//   id: "unique_raga_id",
//   name: "raga_name", 
//   tradition: "carnatic|hindustani",
//   melakarta_number: 1-72, // for Carnatic melakarta ragas
//   thaat: "thaat_name", // for Hindustani ragas
//   arohana: ["S", "R1", "G2", "M1", "P", "D1", "N2", "S'"],
//   avarohana: ["S'", "N2", "D1", "P", "M1", "G2", "R1", "S"],
//   vadi: "primary_swara",
//   samvadi: "secondary_swara", 
//   time_of_day: "morning|afternoon|evening|night",
//   season: "spring|summer|monsoon|autumn|winter",
//   mood: ["devotional", "heroic", "peaceful"],
//   confidence: 0.95, // expert confidence in raga identification
//   verified: true,
//   created_date: "2024-01-01",
//   last_updated: "2024-01-01"
// })

// Artist nodes - represent performers
CREATE CONSTRAINT artist_id_unique IF NOT EXISTS FOR (a:Artist) REQUIRE a.id IS UNIQUE;

// (:Artist {
//   id: "unique_artist_id",
//   name: "artist_name",
//   tradition: "carnatic|hindustani|both",
//   instrument: "primary_instrument",
//   gharana: "gharana_name", // for Hindustani
//   style: "performance_style",
//   birth_year: 1950,
//   active_period: "1970-2020"
// })

// Composition nodes - represent musical pieces
CREATE CONSTRAINT composition_id_unique IF NOT EXISTS FOR (c:Composition) REQUIRE c.id IS UNIQUE;

// (:Composition {
//   id: "unique_composition_id", 
//   title: "composition_title",
//   composer: "composer_name",
//   tradition: "carnatic|hindustani",
//   type: "kriti|varnam|thillana|bandish|khayal",
//   tala: "tala_name",
//   language: "sanskrit|telugu|tamil|hindi|urdu",
//   created_date: "2024-01-01"
// })

// =============================================================================
// RELATIONSHIP TYPES AND PROPERTIES
// =============================================================================

// Cross-tradition raga relationships
// (:Raga)-[:IDENTICAL {confidence: 0.95, verified_by: "expert_id", notes: "..."}]->(:Raga)
// (:Raga)-[:SIMILAR {confidence: 0.85, differences: ["..."], notes: "..."}]->(:Raga)  
// (:Raga)-[:RELATED {confidence: 0.75, relationship_type: "family", notes: "..."}]->(:Raga)
// (:Raga)-[:DERIVED_FROM {confidence: 0.90, evolution_path: "...", notes: "..."}]->(:Raga)

// Performance relationships
// (:Artist)-[:PERFORMED {date: "2024-01-01", venue: "...", quality: 0.9}]->(:Composition)
// (:Composition)-[:IN_RAGA]->(:Raga)
// (:Artist)-[:SPECIALIZES_IN {expertise_level: "master", years_experience: 30}]->(:Raga)

// Structural relationships
// (:Raga)-[:HAS_JANYA {derivation_type: "vakra|varjya|upanga"}]->(:Raga)
// (:Raga)-[:BELONGS_TO_MELAKARTA]->(:Raga)
// (:Raga)-[:SHARES_THAAT]->(:Raga)

// =============================================================================
// INDEXES FOR PERFORMANCE
// =============================================================================

CREATE INDEX raga_tradition_index IF NOT EXISTS FOR (r:Raga) ON (r.tradition);
CREATE INDEX raga_melakarta_index IF NOT EXISTS FOR (r:Raga) ON (r.melakarta_number);
CREATE INDEX raga_thaat_index IF NOT EXISTS FOR (r:Raga) ON (r.thaat);
CREATE INDEX artist_tradition_index IF NOT EXISTS FOR (a:Artist) ON (a.tradition);
CREATE INDEX composition_type_index IF NOT EXISTS FOR (c:Composition) ON (c.type);

// =============================================================================
// USEFUL QUERY PATTERNS (COMMENTED - WILL WORK AFTER DATA POPULATION)
// =============================================================================

// Find all ragas identical to a given raga
// MATCH (r:Raga {name: $raga_name})-[rel:IDENTICAL]-(related)
// RETURN related.name, rel.confidence, rel.notes

// Find raga derivation chains
// MATCH path = (parent:Raga)-[:DERIVED_FROM*]->(derived:Raga)
// WHERE parent.name = $parent_raga_name
// RETURN path

// Find artists who perform both traditions
// MATCH (a:Artist)-[:PERFORMED]->(c1:Composition)-[:IN_RAGA]->(r1:Raga),
//       (a)-[:PERFORMED]->(c2:Composition)-[:IN_RAGA]->(r2:Raga)
// WHERE r1.tradition = "carnatic" AND r2.tradition = "hindustani"
// RETURN DISTINCT a.name

// Find similar ragas across traditions
// MATCH (c:Raga {tradition: "carnatic"})-[rel:SIMILAR]-(h:Raga {tradition: "hindustani"})
// RETURN c.name, h.name, rel.confidence, rel.differences