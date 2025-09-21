-- RagaSense-Data Database Schema for Vercel Postgres
-- This schema supports the collaborative raga mapping system with Ramanarunachalam and Saraga data

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Ragas table (from both Ramanarunachalam and Saraga)
CREATE TABLE ragas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    tradition VARCHAR(50) NOT NULL CHECK (tradition IN ('carnatic', 'hindustani', 'both')),
    source VARCHAR(50) NOT NULL CHECK (source IN ('ramanarunachalam', 'saraga', 'both')),
    arohana TEXT,
    avarohana TEXT,
    melakarta VARCHAR(100),
    janya_type VARCHAR(100),
    song_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Artists table
CREATE TABLE artists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    tradition VARCHAR(50) NOT NULL CHECK (tradition IN ('carnatic', 'hindustani', 'both')),
    source VARCHAR(50) NOT NULL CHECK (source IN ('ramanarunachalam', 'saraga', 'both')),
    song_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Songs table
CREATE TABLE songs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    raga_id UUID REFERENCES ragas(id) ON DELETE CASCADE,
    artist_id UUID REFERENCES artists(id) ON DELETE CASCADE,
    composer VARCHAR(255),
    tradition VARCHAR(50) NOT NULL CHECK (tradition IN ('carnatic', 'hindustani', 'both')),
    source VARCHAR(50) NOT NULL CHECK (source IN ('ramanarunachalam', 'saraga', 'both')),
    youtube_links TEXT[],
    audio_features JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cross-tradition mappings (collaborative)
CREATE TABLE cross_tradition_mappings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    saraga_raga_id UUID REFERENCES ragas(id) ON DELETE CASCADE,
    ramanarunachalam_raga_id UUID REFERENCES ragas(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL CHECK (relationship_type IN ('exact_match', 'similar_match', 'related')),
    similarity_score DECIMAL(3,2) CHECK (similarity_score >= 0 AND similarity_score <= 1),
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    notes TEXT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Collaborative mapping proposals
CREATE TABLE mapping_proposals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    saraga_raga VARCHAR(255) NOT NULL,
    ramanarunachalam_raga VARCHAR(255) NOT NULL,
    proposed_by VARCHAR(255) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    notes TEXT,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Votes on mapping proposals
CREATE TABLE mapping_votes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_id UUID REFERENCES mapping_proposals(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    vote_type VARCHAR(20) NOT NULL CHECK (vote_type IN ('approve', 'reject')),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(proposal_id, user_id)
);

-- Users table for tracking contributions
CREATE TABLE users (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    contributions JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_ragas_name ON ragas(name);
CREATE INDEX idx_ragas_tradition ON ragas(tradition);
CREATE INDEX idx_ragas_source ON ragas(source);
CREATE INDEX idx_artists_name ON artists(name);
CREATE INDEX idx_artists_tradition ON artists(tradition);
CREATE INDEX idx_songs_title ON songs(title);
CREATE INDEX idx_songs_raga_id ON songs(raga_id);
CREATE INDEX idx_songs_artist_id ON songs(artist_id);
CREATE INDEX idx_cross_tradition_mappings_saraga ON cross_tradition_mappings(saraga_raga_id);
CREATE INDEX idx_cross_tradition_mappings_ramanarunachalam ON cross_tradition_mappings(ramanarunachalam_raga_id);
CREATE INDEX idx_mapping_proposals_status ON mapping_proposals(status);
CREATE INDEX idx_mapping_votes_proposal_id ON mapping_votes(proposal_id);
CREATE INDEX idx_mapping_votes_user_id ON mapping_votes(user_id);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_ragas_updated_at BEFORE UPDATE ON ragas FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_artists_updated_at BEFORE UPDATE ON artists FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_songs_updated_at BEFORE UPDATE ON songs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cross_tradition_mappings_updated_at BEFORE UPDATE ON cross_tradition_mappings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mapping_proposals_updated_at BEFORE UPDATE ON mapping_proposals FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

