# GitHub Setup Guide
## Getting the Minsky Project Online and Shareable

Follow these steps in order. Takes about 15 minutes.

---

## Step 1 — Create a GitHub account (skip if you have one)

1. Go to https://github.com
2. Click **Sign up**
3. Use your Marymount email address
4. Choose a username (e.g. `ghaughton-dsc` or similar)
5. Verify your email

---

## Step 2 — Create the repository

1. Once logged in, click the **+** icon (top right) → **New repository**
2. Fill in:
   - **Repository name:** `minsky-motor-hrl`
   - **Description:** `Hierarchical RL framework for robotic manipulation — Minsky Motor Abstractions (ICCWS 2026)`
   - **Visibility:** Public *(required for submission link)*
   - ✅ Check **Add a README file** — uncheck this, we have our own
   - Leave everything else as default
3. Click **Create repository**

You now have an empty repo at:
```
https://github.com/YOUR_USERNAME/minsky-motor-hrl
```

---

## Step 3 — Install Git on your computer (skip if installed)

**Windows:**
1. Download from https://git-scm.com/download/win
2. Install with default settings
3. Open **Git Bash** (search for it in Start menu)

**Mac:**
```bash
# In Terminal
xcode-select --install
```

---

## Step 4 — Configure Git (one time only)

Open Git Bash (Windows) or Terminal (Mac) and run:

```bash
git config --global user.name "Gregory Haughton"
git config --global user.email "your-marymount-email@marymount.edu"
```

---

## Step 5 — Push the project to GitHub

Navigate to your project folder and run these commands one by one:

```bash
# Go to your project folder
cd C:\Users\grego\OneDrive\Desktop\minsky_project

# Initialise git
git init

# Add your GitHub repo as the remote origin
git remote add origin https://github.com/YOUR_USERNAME/minsky-motor-hrl.git

# Stage all files
git add .

# First commit
git commit -m "Initial commit: Minsky Motor Abstractions Phase 4 validated (100% success rate)"

# Push to GitHub
git branch -M main
git push -u origin main
```

> **If Git asks for credentials:** Use your GitHub username and a Personal Access Token (not your password).
> To create a token: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token → check `repo` scope → copy the token and use it as your password.

---

## Step 6 — Verify the upload

1. Go to `https://github.com/YOUR_USERNAME/minsky-motor-hrl`
2. You should see all your files listed
3. The README.md will display automatically on the page

Your shareable link is:
```
https://github.com/YOUR_USERNAME/minsky-motor-hrl
```

---

## Step 7 — Add team members as collaborators

1. Go to your repo on GitHub
2. Click **Settings** (top tab)
3. Click **Collaborators** (left sidebar)
4. Click **Add people**
5. Enter each team member's GitHub username or email
6. They will receive an email invitation — once accepted, they can view and edit the repo

---

## Step 8 — Handle large model files (if push fails)

The `.pth` model files are ~160KB each, which is under GitHub's 100MB limit, so they should push fine. If you get a size error, install Git LFS:

```bash
# Install Git LFS
git lfs install

# Track .pth files with LFS
git lfs track "*.pth"
git add .gitattributes
git commit -m "Add Git LFS tracking for model weights"
git push
```

---

## Step 9 — Keep the repo updated

Any time you make changes to your code:

```bash
git add .
git commit -m "Brief description of what changed"
git push
```

---

## Your Final Submission Link

Once pushed, your submission URL for Canvas will be:
```
https://github.com/YOUR_USERNAME/minsky-motor-hrl
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Quick Reference — Common Git Commands

| Command | What it does |
|---|---|
| `git status` | See what files have changed |
| `git add .` | Stage all changes |
| `git commit -m "message"` | Save a snapshot |
| `git push` | Upload to GitHub |
| `git pull` | Download latest from GitHub |
| `git log --oneline` | See commit history |
