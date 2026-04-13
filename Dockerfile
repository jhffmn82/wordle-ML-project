# Deploying to Hugging Face Spaces

## Step 1: Save curated word list (in your Colab notebook)

Add this cell after building curated lists (Part 2) to avoid a 400s rebuild on every server start:

```python
import pickle
with open("models/curated_set3.pkl", "wb") as f:
    pickle.dump(set3, f)
print(f"Saved {len(set3)} curated words to models/curated_set3.pkl")
```

Then push to GitHub.

## Step 2: Add web app files to your repo

Copy these files into your existing repo:

```
wordle-ML-project/
├── web/
│   ├── app.py               ← NEW (replace old Dash app)
│   ├── solver_adapter.py     ← NEW
│   └── templates/
│       └── index.html        ← NEW
├── Dockerfile                ← NEW
├── README.md                 ← NEW (HF Spaces metadata)
├── requirements.txt          ← UPDATE (add flask, gunicorn)
└── ... (existing engine/, solvers/, models/)
```

## Step 3: Create the Space

1. Go to https://huggingface.co/new-space
2. Choose a name (e.g., `wordle-ml`)
3. Select **Docker** as the SDK
4. Select **Public** visibility
5. Click **Create Space**

## Step 4: Push your code

```bash
# Add HF Spaces as a second remote
cd wordle-ML-project
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/wordle-ml

# Push
git push hf main
```

HF will build the Docker image and deploy. The app will be live at:
`https://YOUR_USERNAME-wordle-ml.hf.space`

## Step 5: Verify

- Home page loads with 3 navigation cards
- Solver Assistant: select a solver, type a word, click tiles, get suggestions
- Autoplay: enter a target word, watch the solver play
- About: performance table and charts display

## Troubleshooting

- **"Solver not available"**: Model files (`.pt`, `.pkl`) not found. Make sure `models/` is pushed.
- **Slow first load**: The infogain solver computes its opening guess (~30s). Subsequent requests are fast.
- **TabularQ not loading**: You need `models/curated_set3.pkl`. See Step 1.
- **Build fails on torch**: The Dockerfile installs CPU-only torch. If you see CUDA errors, the solver code may need a `map_location='cpu'` in `torch.load()`.

## Notes

- The app uses port 7860 (HF Spaces default for Docker)
- Solvers load lazily on first request — any solver that fails to import just shows as unavailable
- The rollout solver loads its cache from `models/rollout_cache.pkl` (431 states)
- DQN v2 will appear automatically when `models/dqn_v2_model.pt` exists
