-- =============================================================================
-- Taiwan Framing Annotation (L1 / L2) — Supabase schema
-- Run in Supabase SQL Editor. RLS off by default; app filters by annotator_id
-- for writes; peer notes are readable by all authenticated app users (anon key).
-- =============================================================================

CREATE TABLE IF NOT EXISTS framing_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    annotator_id TEXT NOT NULL,
    utterance_id INTEGER NOT NULL,
    l1_label TEXT NOT NULL,
    l2_labels TEXT DEFAULT '',
    unsure BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(annotator_id, utterance_id)
);

CREATE INDEX IF NOT EXISTS idx_framing_ann_annotator ON framing_annotations(annotator_id);
CREATE INDEX IF NOT EXISTS idx_framing_ann_utterance ON framing_annotations(utterance_id);

-- Short peer notes for the same utterance (互助討論)
CREATE TABLE IF NOT EXISTS framing_peer_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    utterance_id INTEGER NOT NULL,
    annotator_id TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_framing_notes_utterance ON framing_peer_notes(utterance_id);
