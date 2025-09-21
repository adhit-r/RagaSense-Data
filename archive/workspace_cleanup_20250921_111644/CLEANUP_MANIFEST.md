# Workspace Cleanup Manifest
## Date: 2025-09-21 11:16:44

### Files Moved to Archive

**Redundant Applications:**
- `advanced_raga_mapper_backup.py` → `archive/workspace_cleanup_20250921_111644/`
- `advanced_raga_mapper.py` → `archive/workspace_cleanup_20250921_111644/` (too complex, overlapping functionality)
- `app_with_database.py` → `archive/workspace_cleanup_20250921_111644/`
- `simple_raga_mapper_improved.py` → `archive/workspace_cleanup_20250921_111644/`

**Utility Scripts:**
- `cleanup_root_directory.py` → `archive/workspace_cleanup_20250921_111644/`
- `docker-compose-postgresql.yml` → `archive/workspace_cleanup_20250921_111644/`

### Rationale

**Application Consolidation:**
- Keeping `app.py` as the main collaborative mapping application with voting system
- Keeping `modern_raga_mapper.py` as the clean UI alternative for individual mapping
- Archived `advanced_raga_mapper.py` due to complexity overlap and maintainability concerns

**Clean Root Directory:**
- Removed backup and development files
- Removed utility scripts that are no longer needed
- Focused on core applications and configuration files

### Files Remaining in Root

**Core Applications:**
- `app.py` - Main collaborative raga mapper
- `modern_raga_mapper.py` - Clean UI raga mapper

**Configuration:**
- `requirements.txt`
- `vercel.json`
- `.gitignore`
- `.vercelignore`

**Documentation:**
- `README.md`
- `WORKSPACE_ORGANIZATION_PLAN.md`

**Directories:**
- `config/` - Project configuration
- `data/` - Dataset (6-tier structure)
- `database/` - Database schemas and migration
- `deployment/` - Deployment configurations
- `docs/` - Documentation
- `scripts/` - Processing scripts
- `website/` - Astro website
- `archive/` - Archived files

This cleanup prepares the workspace for GitHub while maintaining the core functionality needed for raga mapping and research.