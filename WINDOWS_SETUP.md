# Windows Setup Guide (Git Bash / MINGW64)

## Quick Fix for Your Current Issue

You're almost there! The issue is Windows path separators. Here's how to fix:

### Option 1: Use Python Directly (Recommended)

Instead of using the broken `setup.py`, run commands manually:

```bash
# 1. You already have venv created ✓
# Activate it (Windows Git Bash):
source venv/Scripts/activate

# 2. Install dependencies (use Python directly, not pip path)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Update .env file
notepad .env
# Add your API keys:
# THE_ODDS_API_KEY=6430e56502091b0dc2d77f1c77745cb2
# KENPOM_API_KEY=your_key_here
# DATABASE_URL=postgresql://postgres:LzI8_0MqxEQR5RKdrSBXqQ@localhost:5432/cbb_edge

# 4. Initialize database
python scripts/init_db.py --seed

# 5. Run tests
python -m pytest tests/ -v
```

### Option 2: Use Fixed Setup Script

```bash
# Use the Windows-compatible version
python setup_windows.py
```

---

## Common Windows Issues & Fixes

### Issue 1: "venv/bin/activate: No such file or directory"

**Windows uses `Scripts` not `bin`:**

```bash
# ❌ Wrong (Linux/Mac):
source venv/bin/activate

# ✅ Correct (Windows Git Bash):
source venv/Scripts/activate

# ✅ Also works (Windows CMD):
venv\Scripts\activate.bat

# ✅ Also works (Windows PowerShell):
venv\Scripts\Activate.ps1
```

### Issue 2: "pip install failed"

**Use Python module syntax instead of pip path:**

```bash
# ❌ Wrong:
venv\Scripts\pip install package

# ✅ Correct:
python -m pip install package
```

### Issue 3: Docker PostgreSQL

**Your Docker container is already running!** ✓

Database connection string:
```
postgresql://postgres:LzI8_0MqxEQR5RKdrSBXqQ@localhost:5432/cbb_edge
```

Test connection:
```bash
# Install psycopg2 if needed
python -m pip install psycopg2-binary

# Test connection
python -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:LzI8_0MqxEQR5RKdrSBXqQ@localhost:5432/cbb_edge'); print('✅ Connected!')"
```

---

## Complete Manual Setup (Windows)

If automated script fails, here's the full manual process:

### Step 1: Virtual Environment

```bash
# Create (already done ✓)
python -m venv venv

# Activate
source venv/Scripts/activate

# Verify
which python
# Should show: ~/repos/v1/cbb-edge/venv/Scripts/python
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all packages
python -m pip install -r requirements.txt

# This will install:
# - fastapi, uvicorn, sqlalchemy
# - numpy, scipy, pandas
# - requests, beautifulsoup4
# - streamlit, plotly
# - pytest
# Total: ~25 packages, takes 2-3 minutes
```

### Step 3: Configure .env

```bash
# Copy template
cp .env.example .env

# Edit with notepad
notepad .env
```

Add these values:
```bash
DATABASE_URL=postgresql://postgres:LzI8_0MqxEQR5RKdrSBXqQ@localhost:5432/cbb_edge
THE_ODDS_API_KEY=6430e56502091b0dc2d77f1c77745cb2
KENPOM_API_KEY=your_kenpom_key_here
API_KEY_USER1=Xt7fqimHr7D-mfBCliQLcTMHWdlNx2KuZ8Uhgs_4hIE
```

### Step 4: Initialize Database

```bash
# Run init script
python scripts/init_db.py --seed

# Expected output:
# ✅ Database tables created
# ✅ Test data seeded
```

If this fails, check:
```bash
# Is PostgreSQL running?
docker ps | grep cbb-postgres

# If not, start it:
docker start cbb-postgres

# Or create new container:
docker run -d --name cbb-postgres \
  -e POSTGRES_PASSWORD=LzI8_0MqxEQR5RKdrSBXqQ \
  -e POSTGRES_DB=cbb_edge \
  -p 5432:5432 \
  postgres:15
```

### Step 5: Run Tests

```bash
# Run test suite
python -m pytest tests/ -v

# Expected: 15 tests pass
# test_betting_model.py::TestMonteCarloCI::test_returns_three_values PASSED
# ... (15 total)
```

### Step 6: Start Backend

```bash
# Start FastAPI server
python -m uvicorn backend.main:app --reload --port 8000

# Expected output:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete

# Test it:
curl http://localhost:8000/health
# Or open in browser: http://localhost:8000
```

### Step 7: Start Dashboard (New Window)

Open a **new** Git Bash window:

```bash
# Navigate to project
cd ~/repos/v1/cbb-edge

# Activate venv
source venv/Scripts/activate

# Set environment variables
export API_URL=http://localhost:8000
export API_KEY=Xt7fqimHr7D-mfBCliQLcTMHWdlNx2KuZ8Uhgs_4hIE

# Start Streamlit
python -m streamlit run dashboard/app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

---

## Verification Checklist

After setup, verify everything works:

### ✅ Backend Health Check

```bash
curl http://localhost:8000/health

# Expected:
{
  "status": "healthy",
  "database": "connected",
  "scheduler": "running",
  "timestamp": "2026-02-11T..."
}
```

### ✅ API Endpoints

```bash
# Set API key
export API_KEY=Xt7fqimHr7D-mfBCliQLcTMHWdlNx2KuZ8Uhgs_4hIE

# Test predictions endpoint
curl http://localhost:8000/api/predictions/today \
  -H "X-API-Key: $API_KEY"

# Expected (first time): 
{
  "date": "2026-02-11",
  "total_games": 0,
  "bets_recommended": 0,
  "predictions": []
}
```

### ✅ Dashboard Access

Open browser: http://localhost:8501

Should show:
- Dashboard header
- "Total Bets: 0"
- Performance section (empty)
- No errors

### ✅ Database Connection

```bash
# Connect to database
docker exec -it cbb-postgres psql -U postgres -d cbb_edge

# Run queries:
\dt  # List tables
SELECT COUNT(*) FROM games;  # Should be 0
\q  # Quit
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'fastapi'"

```bash
# Ensure venv is activated
source venv/Scripts/activate

# Verify Python path
which python
# Should show venv path, not system Python

# Reinstall dependencies
python -m pip install -r requirements.txt
```

### "Database connection failed"

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check connection string in .env
cat .env | grep DATABASE_URL

# Test connection
python -c "from backend.models import engine; engine.connect(); print('✅ Connected')"
```

### "Port 8000 already in use"

```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill it (use PID from above)
taskkill /PID <pid> /F

# Or use different port
python -m uvicorn backend.main:app --reload --port 8001
```

### "Docker command not found"

**Install Docker Desktop for Windows:**
- Download: https://www.docker.com/products/docker-desktop/
- Install and restart
- Run `docker --version` to verify

**Alternative (no Docker):**
- Install PostgreSQL directly: https://www.postgresql.org/download/windows/
- Use connection string: `postgresql://postgres:password@localhost:5432/cbb_edge`

---

## Quick Reference (Windows Commands)

```bash
# Activate venv (Git Bash)
source venv/Scripts/activate

# Activate venv (CMD)
venv\Scripts\activate.bat

# Activate venv (PowerShell)
venv\Scripts\Activate.ps1

# Deactivate venv
deactivate

# Install package
python -m pip install package_name

# Run Python script
python script.py

# Run tests
python -m pytest tests/ -v

# Start backend
python -m uvicorn backend.main:app --reload

# Start dashboard
python -m streamlit run dashboard/app.py

# View logs
tail -f logs/app.log  # Git Bash
# Or use: Get-Content logs/app.log -Wait  # PowerShell
```

---

## Next Steps After Setup

Once everything is running:

1. **Trigger first analysis:**
   ```bash
   curl -X POST http://localhost:8000/admin/run-analysis \
     -H "X-API-Key: Xt7fqimHr7D-mfBCliQLcTMHWdlNx2KuZ8Uhgs_4hIE"
   ```

2. **Check results in dashboard:**
   - Open http://localhost:8501
   - View "Today's Bets" tab

3. **Review predictions:**
   ```bash
   curl http://localhost:8000/api/predictions/today \
     -H "X-API-Key: Xt7fqimHr7D-mfBCliQLcTMHWdlNx2KuZ8Uhgs_4hIE" | jq
   ```

4. **Read documentation:**
   - README.md - Project overview
   - IMPLEMENTATION.md - What to do next

---

## Git Bash Tips

```bash
# Create aliases for convenience
echo "alias activate='source venv/Scripts/activate'" >> ~/.bashrc
echo "alias backend='python -m uvicorn backend.main:app --reload'" >> ~/.bashrc
echo "alias dashboard='python -m streamlit run dashboard/app.py'" >> ~/.bashrc
source ~/.bashrc

# Now you can just run:
activate
backend  # In one window
dashboard  # In another window
```

---

Need help? Check:
- README.md - Full docs
- QUICKREF.md - Command reference
- INSTALL.md - Detailed installation guide
