# Data

This folder contains the metadata CSV files used by the notebooks. The raw datasets (images and weather CSV) are **not tracked in Git** due to their size — follow the download instructions below to set up your local environment.

---

## Tracked Files (in this repo)

| File | Size | Description |
|---|---|---|
| `train_image_class.csv` | Small | Image filenames + labels for training (Flickr8k, 5 classes) |
| `valid_image_class.csv` | Small | Image filenames + labels for validation (Flickr8k, 5 classes) |

---

## Dataset Downloads

### Part 1 — Flickr8k Image Dataset

| Field | Details |
|---|---|
| **Download link** | https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip |
| **Format** | ZIP archive of ~8,000 JPEG images |
| **Size** | ~1 GB |

**Setup steps:**
1. Download and unzip `Flickr8k_Dataset.zip`
2. Place all extracted `.jpg` image files directly inside this `data/` folder
3. Your folder should look like this:

```
data/
├── train_image_class.csv     ← already here (tracked)
├── valid_image_class.csv     ← already here (tracked)
├── 1000268201_693b08cb0e.jpg
├── 1001773457_577c3a7d70.jpg
├── ...                       ← ~8,000 more .jpg files
```

> The image files are listed in `.gitignore` and will not be committed to Git.

---

### Part 2 — London Weather Dataset

| Field | Details |
|---|---|
| **Download link** | https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data |
| **Format** | CSV |
| **Size** | Small (~500 KB) |

**Setup steps:**
1. Download `london_weather.csv` from Kaggle (requires a free Kaggle account)
2. Place it directly inside this `data/` folder:

```
data/
├── london_weather.csv        ← download and place here
├── train_image_class.csv
├── valid_image_class.csv
```

> `london_weather.csv` is listed in `.gitignore` and will not be committed to Git.

**Alternatively**, in the Part 2 notebook the file is loaded from Google Drive. If you are running in Colab, mount your Drive and update the file path in the notebook accordingly.

---

## CSV Format Reference

### `train_image_class.csv` and `valid_image_class.csv`

Used by Part 1 (`Team 8_Part1_Notebook.ipynb`) to map image filenames to their category labels.

| Column | Type | Description |
|---|---|---|
| `Image Path` | string | Filename of the image (e.g. `1000268201_693b08cb0e.jpg`) |
| `Label` | string | One of: `animals`, `objects`, `people`, `scenes`, `others` |

These files are loaded via `ImageDataGenerator.flow_from_dataframe()` with `directory` pointing to this `data/` folder.

### `london_weather.csv`

Used by Part 2 (`Team 8_Part2_Notebook.ipynb`) for temperature forecasting.

| Column | Type | Description |
|---|---|---|
| `date` | int (YYYYMMDD) | Observation date |
| `mean_temp` | float | Mean daily temperature (°C) |
| *(other columns)* | various | Ignored by the notebook; only `date` and `mean_temp` are used |

---

## Quick Checklist Before Running the Notebooks

- [ ] Flickr8k `.jpg` images extracted into `data/`
- [ ] `train_image_class.csv` present in `data/`
- [ ] `valid_image_class.csv` present in `data/`
- [ ] `london_weather.csv` downloaded and placed in `data/` (or accessible via Google Drive)
