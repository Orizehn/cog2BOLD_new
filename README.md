# fMRI Activation Maps Viewer

Interactive Streamlit application for exploring fMRI activation maps based on cognitive features and stimuli.

## 🚀 Quick Start

### For Users

**Visit the live app:** [Your Streamlit App URL will be here after deployment]

The app will automatically download required data files from Google Drive on first run.

### For Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app_actmaps.py
   ```

## 📦 Setup for Deployment

### Step 1: Prepare Your Data Files on Google Drive

You need to upload these files to Google Drive and make them publicly accessible:

**Required files:**
- `ridge_actmaps_stimXcog.pkl` - Mean activation model weights
- `ridge_var_actmaps_stimXcog.pkl` - Variance model weights
- `DFG_behavioural_results.csv` - Behavioral data
- At least one `session_*_cog2bold.pkl` file (e.g., `session_1_cog2bold.pkl`)

**Steps to upload and get file IDs:**

1. Upload each file to Google Drive
2. Right-click on the file → **Share**
3. Click **Change to anyone with the link** → Set to **Viewer**
4. Click **Copy link**
5. Extract the FILE_ID from the URL:
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
                                    ^^^^^^^^^^^^
   ```

### Step 2: Update File IDs in Code

Edit `app_actmaps.py` and update the `GDRIVE_FILES` dictionary (around line 24):

```python
GDRIVE_FILES = {
    "ridge_actmaps_stimXcog.pkl": "1abc123def456ghi789jkl",  # Replace with your file ID
    "ridge_var_actmaps_stimXcog.pkl": "1xyz789uvw456rst123",
    "DFG_behavioural_results.csv": "1mno456pqr789stu123vwx",
    "session_1_cog2bold.pkl": "1def789ghi123jkl456mno",
    # Add more session files if needed
}
```

### Step 3: Push to GitHub

1. **Initialize git repository (if not already done):**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create a new repository on GitHub:**
   - Go to [github.com/new](https://github.com/new)
   - Create a new repository (public or private)
   - **Don't** initialize with README (we already have files)

3. **Push your code:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 4: Deploy on Streamlit Community Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Configure deployment:**
   - Repository: Select your repository
   - Branch: `main`
   - Main file path: `app_actmaps.py`
5. **Click "Deploy"**

The app will build and deploy automatically. First-time users will see the data download process.

## 📁 Project Structure

```
fmri_activation_viewer_deploy/
├── app_actmaps.py           # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore             # Files to exclude from git
└── DFG_data/              # Data directory (created automatically)
    └── dnn_datasets/      # Model files (downloaded on first run)
```

## 🎛️ Features

- **Three analysis modes:**
  - Single stimulus activation maps
  - Variance maps
  - Statistical contrasts between stimuli

- **Interactive cognitive feature sliders:**
  - Adjust cognitive parameters in real-time
  - Reset to global or session-specific means

- **Flexible time aggregation:**
  - HRF-weighted averaging
  - Individual TR viewing
  - Custom weighted averaging
  - Maximum activation across TRs

- **Visualization controls:**
  - Adjustable color scale
  - Auto-scaling to data range
  - Configurable slice selection

## 🔧 Troubleshooting

### "Failed to download file"
- Ensure the Google Drive link is set to "Anyone with the link can view"
- Verify the file ID is correct
- Check your internet connection

### "No session_*_cog2bold.pkl found"
- Make sure you've added at least one session pickle file to GDRIVE_FILES
- Verify the file name matches the pattern `session_*_cog2bold.pkl`

### App crashes or shows errors
- Check Streamlit Cloud logs (click "Manage app" → "Logs")
- Verify all required files were downloaded successfully
- Ensure file formats match expected structure

## 📊 Data Requirements

The app expects specific data formats:

**Model files (.pkl):**
- Must contain `W` (weights), `cog_base_cols`, `stimulus_labels`, `n_stimuli`, `cog_mean`, `cog_std`

**Session files (.pkl):**
- Must contain `meta` dictionary with `masker` object

**Behavioral CSV:**
- Must have column specified in `SUBJECT_COL` (default: "Subject")
- Cognitive features as columns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

[Add your license here]

## 👥 Authors

[Add your name and contact information]

## 🙏 Acknowledgments

[Add acknowledgments]
