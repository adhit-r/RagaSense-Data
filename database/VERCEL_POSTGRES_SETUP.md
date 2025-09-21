# Vercel Postgres Setup Guide for RagaSense-Data

This guide will help you set up Vercel Postgres and connect it to your RagaSense-Data collaborative mapping system.

## 🚀 Step 1: Create Vercel Postgres Database

### Option A: Via Vercel Dashboard
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project or create a new one
3. Go to **Storage** tab
4. Click **Create Database** → **Postgres**
5. Choose a name (e.g., `ragasense-db`)
6. Select a region close to your users
7. Click **Create**

### Option B: Via Vercel CLI
```bash
vercel storage create postgres --name ragasense-db
```

## 🔧 Step 2: Get Database Connection Details

After creating the database, you'll get connection details:

```bash
# Get connection string
vercel storage connect postgres ragasense-db

# Or get individual environment variables
vercel env pull .env.local
```

Your `.env.local` should contain:
```env
POSTGRES_HOST=your-host.vercel-storage.com
POSTGRES_DATABASE=ragasense-db
POSTGRES_USER=your-username
POSTGRES_PASSWORD=your-password
POSTGRES_PORT=5432
```

## 📊 Step 3: Set Up Database Schema

### Install Dependencies
```bash
pip install psycopg2-binary python-dotenv
```

### Create Database Schema
```bash
# Connect to your database and run the schema
psql "postgresql://username:password@host:port/database" -f database/schema.sql
```

Or use the Python script:
```bash
python database/setup_schema.py
```

## 📦 Step 4: Migrate Your Data

### Run Data Migration
```bash
# Set environment variables
export POSTGRES_HOST="your-host.vercel-storage.com"
export POSTGRES_DATABASE="ragasense-db"
export POSTGRES_USER="your-username"
export POSTGRES_PASSWORD="your-password"
export POSTGRES_PORT="5432"

# Run migration
python database/migrate_data.py
```

This will migrate:
- ✅ All ragas from your unified dataset
- ✅ Artists and songs data
- ✅ Existing cross-tradition mappings
- ✅ Collaborative mapping proposals (if any)

## 🌐 Step 5: Deploy to Vercel

### Update Your Vercel Configuration

Create/update `vercel.json`:
```json
{
  "buildCommand": "pip install -r requirements.txt",
  "outputDirectory": ".",
  "framework": "flask",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/app_with_database.py"
    }
  ],
  "env": {
    "POSTGRES_HOST": "@postgres-host",
    "POSTGRES_DATABASE": "@postgres-database", 
    "POSTGRES_USER": "@postgres-user",
    "POSTGRES_PASSWORD": "@postgres-password",
    "POSTGRES_PORT": "@postgres-port"
  }
}
```

### Update Requirements
Add to `requirements.txt`:
```
Flask==2.3.3
psycopg2-binary==2.9.7
flask-cors==4.0.0
python-dotenv==1.0.0
```

### Deploy
```bash
vercel --prod
```

## 🔍 Step 6: Verify Database Connection

### Test the API
```bash
# Test ragas endpoint
curl https://your-app.vercel.app/api/ragas

# Test collaborative mapping
curl https://your-app.vercel.app/api/collaborative/pending
```

### Check Database
```bash
# Connect to database
vercel storage connect postgres ragasense-db

# Check tables
\dt

# Check ragas count
SELECT COUNT(*) FROM ragas;

# Check cross-tradition mappings
SELECT COUNT(*) FROM cross_tradition_mappings;
```

## 📈 Step 7: Monitor and Maintain

### Database Monitoring
- Monitor usage in Vercel Dashboard → Storage
- Set up alerts for high usage
- Regular backups (Vercel handles this automatically)

### Performance Optimization
- Add indexes for frequently queried columns
- Use connection pooling for high traffic
- Monitor query performance

## 🛠️ Troubleshooting

### Common Issues

**Connection Failed**
```bash
# Check environment variables
vercel env ls

# Test connection
python -c "import psycopg2; print('Connection test')"
```

**Migration Errors**
```bash
# Check data files exist
ls -la data/04_ml_datasets/unified/

# Run with debug
python database/migrate_data.py --debug
```

**API Errors**
```bash
# Check logs
vercel logs

# Test locally
python app_with_database.py
```

## 📊 Database Schema Overview

### Tables Created
- `ragas` - All ragas from Ramanarunachalam and Saraga
- `artists` - Artist information
- `songs` - Song metadata with YouTube links
- `cross_tradition_mappings` - Existing validated mappings
- `mapping_proposals` - Community proposals
- `mapping_votes` - User votes on proposals
- `users` - User contribution tracking

### Key Features
- ✅ UUID primary keys
- ✅ JSONB metadata fields
- ✅ Automatic timestamps
- ✅ Foreign key relationships
- ✅ Performance indexes
- ✅ Data validation constraints

## 🎯 Next Steps

1. **Test the collaborative mapping interface**
2. **Import your existing mappings**
3. **Set up user authentication** (optional)
4. **Add more data sources**
5. **Implement advanced search features**

## 📞 Support

If you encounter issues:
1. Check Vercel documentation: https://vercel.com/docs/storage/vercel-postgres
2. Review the migration logs
3. Test database connection manually
4. Check environment variables are set correctly

---

**Your RagaSense-Data collaborative mapping system is now connected to a real database! 🎵**

