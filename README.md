# Vegetation Response to Compound Heat Wave and Drought Events

This repository contains the Python-based analysis workflow developed for my Master's thesis at Leipzig University.

The thesis investigates how different vegetation types respond to **compound heat wave and drought (CHD) events** at the global scale. The analysis focuses on events occurring between **2002 and 2019** and examines vegetation response in terms of **resistance, recovery, resilience, and kNDVI anomalies**. In a subsequent modelling step, machine learning is used to investigate environmental and climatic drivers of variability in vegetation response.

---

## Research question

Compound heat wave and drought events represent climate extremes in which high temperatures and water stress occur simultaneously. The objective of this study is to investigate whether vegetation types differ systematically in their response to these compound events and which climatic and environmental conditions help explain this variability.

The analysis therefore addresses two main aspects:

1. **How does vegetation respond to compound heat wave and drought events?**
2. **Which environmental and climatic variables help explain differences in vegetation response?**

---

## Study period and event selection

The analysis is based on a global database of compound heat wave and drought events.

Events are filtered according to:

* event duration of **at least 18 days** in the current analysis implementation;
* event start and end dates falling within the period covered by the kNDVI analysis;
* events occurring between **2002 and 2019**.

After filtering, **189 compound heat wave and drought events** are analysed.

The event information is used to create spatial masks identifying the areas affected by each individual event.

---

## Data

### Earth System Data Cube

The vegetation response and climatic predictors are derived from the **Earth System Data Cube (ESDC)**.

The workflow accesses the ESDC at 0.25° spatial resolution and creates a local subset containing the variables required for the analysis.

The following variables are used:

| Variable                 | Role                         |
| ------------------------ | ---------------------------- |
| `kndvi`                  | Vegetation response variable |
| `air_temperature_2m`     | Mean air temperature         |
| `min_air_temperature_2m` | Minimum air temperature      |
| `max_air_temperature_2m` | Maximum air temperature      |
| `precipitation_era5`     | Precipitation                |
| `root_moisture`          | Root-zone soil moisture      |
| `evaporation`            | Evaporation                  |
| `potential_evaporation`  | Potential evaporation        |

The local ESDC subset is stored as:

```text
data/esdc_subset.zarr
```

The subset is generated for the period **2000–2021** to provide sufficient temporal context around the events.

### Compound heat wave and drought events

The event summary table is downloaded from Zenodo during execution of the workflow.

The event dataset provides information including:

* event labels;
* start and end dates;
* event duration;
* spatial extent;
* heat and drought characteristics;
* compound-event information.

An additional event-label cube is used to identify the spatial footprint of individual events.

### Vegetation data

Vegetation classes are obtained from the GLDAS dominant vegetation dataset.

The workflow additionally uses:

```text
data/GLDASp5_domveg_NOAH3.6_025d.nc4
data/GLDA_veg_legend.csv
```

The vegetation classes are used to aggregate the spatially distributed observations into vegetation-specific time series and statistics.

### Land mask

A GLDAS land mask is used to restrict the event footprints to land areas:

```text
data/GLDASp5_landmask_025d.nc4
```

---

# Analysis workflow

The complete analysis consists of several stages.

<p align="center">
  <img src="figures/1_minicube_creation.png" alt="Creation of minicubes used for modeling" width="800">
</p>

<p align="center">
  <em>Creation of minicubes used for modeling.</em>
</p>

---

This workflow is made more interactive with the notebooks 1 - 6. However, these notebooks are not required for the analysis, they only provide a sandbox environment. The analysis is run by analysis_execution.py.

# 1. Data exploration

[`1_data_exploration.ipynb`](1_data_exploration.ipynb) is used to explore and inspect the datasets required for the analysis.

The notebook:

* loads the compound-event summary data;
* examines the event duration distribution;
* filters events according to duration and study period;
* accesses ESDC data;
* inspects the event label cube;
* examines the land mask;
* loads the GLDAS vegetation classification;
* explores the spatial and temporal characteristics of the input data.

This notebook is primarily exploratory and is not required for the automated processing of all events. The corresponding automated processing of all selected events is implemented in `analysis_execution.py`.

---

# 2. Preprocessing and STL decomposition

[`2_data_preprocessing.ipynb`](2_data_preprocessing.ipynb) investigates the preprocessing required for the vegetation time series.

The main challenge is to separate the vegetation signal associated with an extreme event from the normal seasonal and long-term dynamics of vegetation.

The notebook therefore evaluates **STL (Seasonal-Trend decomposition using Loess)**.

The preprocessing workflow includes:

* sampling locations globally;
* constructing kNDVI time series;
* aligning seasonal cycles between hemispheres;
* testing STL decomposition;
* examining residuals;
* evaluating the behaviour of the decomposition for different types of time series.

The final analysis uses STL residuals to characterise deviations from the expected vegetation dynamics.

This notebook is primarily to play around and is not required for the automated processing of all events. The corresponding automated processing of all selected events is implemented in `analysis_execution.py`.

---

# 3. Event-level aggregation

[`3_aggregation.ipynb`](3_aggregation.ipynb) demonstrates the complete processing workflow for **one individual event**.

For a selected event, the notebook:

1. loads the relevant datasets;
2. creates the spatial event mask;
3. extracts the affected area;
4. extracts kNDVI around the event;
5. applies cosine latitude weighting;
6. aggregates kNDVI separately for vegetation classes;
7. decomposes the resulting time series;
8. calculates vegetation response metrics;
9. aggregates climatic predictors by vegetation class;
10. calculates predictor anomalies;
11. combines the resulting variables into a single dataframe.

The notebook serves primarily as a development and inspection environment.

This notebook is primarily to play around and is not required for the automated processing of all events. The corresponding automated processing of all selected events is implemented in `analysis_execution.py`.

---

# 4. Automated event analysis

[`analysis_execution.py`](analysis_execution.py) contains the main event-processing workflow.

The script loops over all selected compound heat wave and drought events.

For every event, it:

### Event mask

Creates a spatial mask from the event-label cube and restricts the affected area to land.

### kNDVI extraction

Extracts a temporal and spatial subset of kNDVI around the event.

A temporal buffer of **two years** is used for the kNDVI and predictor data cubes.

### Vegetation aggregation

The affected pixels are grouped by vegetation class and aggregated into vegetation-specific time series.

Spatial aggregation uses cosine latitude weighting for mean values.

### Vegetation response

The kNDVI time series are decomposed using STL.

The analysis derives event-level response statistics including:

* `vpre` — pre-event vegetation state;
* `vdist` — vegetation state at maximum disturbance;
* `vpost` — post-event vegetation state;
* `resilience`;
* `resistance`;
* `recovery`;
* `recovery_dep`;
* `n_days_vdist`;
* kNDVI anomaly-related metrics.

### Climatic predictors

For each event and vegetation class, the following predictors are aggregated:

* air temperature;
* minimum air temperature;
* maximum air temperature;
* precipitation;
* root-zone soil moisture;
* evaporation;
* potential evaporation.

The analysis also calculates predictor anomalies relative to the surrounding temporal dynamics.

Previous-year conditions are additionally aggregated to provide information about antecedent environmental conditions.

### Event history

The workflow checks whether other compound events occurred around the analysed event.

This information is stored as the `other_events` variable.

### Output

The resulting event-level data are concatenated into a single dataframe and written to CSV.

The output contains event- and vegetation-class-level response metrics together with climatic predictors and anomalies.

---

# 5. Statistical analysis

[`4_analysis.ipynb`](4_analysis.ipynb) applies the event-processing workflow to the complete event selection and contains the subsequent statistical analysis and visualisation.

The notebook includes analyses of:

* vegetation response distributions;
* resilience;
* resistance;
* recovery;
* anomaly strength;
* differences between vegetation classes;
* relationships between response variables and environmental predictors;
* statistical properties of the resulting time series and residuals.

The computationally intensive event processing is delegated to `analysis_execution.py`.

---

# 6. Dimensionality reduction

[`5_pca.ipynb`](5_pca.ipynb) investigates dimensionality reduction of the predictor dataset. This step was not used in the final analysis, as the dimensionality reduction did not yield meaningful components of the original data (e.g. "hydrological variables"). 

This notebook is primarily to play around and is not required for the automated processing of all events. The corresponding automated processing of all selected events is implemented in `analysis_execution.py`.

Before dimensionality reduction, the predictor variables are standardised using `StandardScaler`.

Two approaches are investigated:

### Principal Component Analysis

PCA is performed using scikit-learn.

The analysis uses five principal components for the investigated representation.

### Isomap

Isomap is additionally used as a nonlinear dimensionality-reduction method.

The resulting embedding is used to investigate the structure of the predictor space.

The dimensionality-reduction analysis is exploratory and supports the interpretation of relationships among the climatic predictors.

---

# 7. Machine learning

[`6_modeling.ipynb`](6_modeling.ipynb) contains the machine-learning analysis step.

The objective is to identify climatic and environmental predictors associated with variability in vegetation response.

Four target variables are modelled:

```text
resilience
resistance
recovery
kNDVI anomaly strength
```

The predictor dataset is constructed by removing response variables, identifiers, vegetation-class information and selected variables that are not used as predictors.

The data are split into:

```text
80 % training data
20 % test data
```

The split is stratified according to vegetation class.

This ensures that the vegetation-class composition is retained between the training and test datasets.

---

## XGBoost

The models use **Extreme Gradient Boosting (XGBoost)** regression.

Separate models are trained for:

* resilience;
* resistance;
* recovery;
* kNDVI anomaly strength.

Hyperparameters are optimised using **Bayesian optimisation with Hyperopt**.

The investigated hyperparameters include:

* learning rate (`eta`);
* `gamma`;
* `max_depth`;
* `min_child_weight`;
* `subsample`;
* `colsample_bytree`;
* L1 regularisation (`alpha`);
* L2 regularisation (`reg_lambda`).

Cross-validation is used during the optimisation process.

This notebook is primarily to play around and is not required for the automated processing of all events. The corresponding automated processing of all selected events is implemented in `analysis_execution.py`.

---

# 8. Model interpretation with SHAP

The trained XGBoost models are interpreted using **SHAP (SHapley Additive exPlanations)**.

SHAP is used to investigate:

* feature importance;
* the contribution of individual predictors;
* the direction of predictor effects;
* relationships between predictor values and model predictions.

The analysis includes SHAP-based visualisations for the different vegetation-response models.

This provides an interpretable link between the machine-learning results and the environmental processes potentially driving vegetation responses.

---

# Utility modules

The `utils/` directory contains reusable functions used throughout the notebooks and automated workflow.

```text
utils/
├── __init__.py
├── anomaly_score.py
├── event_analysis.py
├── event_sampling.py
└── event_visualization.py
```

### `event_sampling.py`

Functions for:

* extracting valid pixel coordinates;
* creating event masks;
* extracting event-specific data cubes;
* spatially cropping data around event areas.

### `event_analysis.py`

Functions for:

* cosine latitude weighting;
* spatial aggregation by vegetation class;
* STL decomposition;
* calculation of vegetation response metrics;
* predictor anomaly calculation;
* calculation of event-related statistics.

### `event_visualization.py`

Functions for generating visualisations of event-level and aggregated time series and spatial data.

### `anomaly_score.py`

Contains additional functions related to anomaly-score calculations used during the preprocessing and analysis development.

---

# Repository structure

```text
masters_thesis/
│
├── 1_data_exploration.ipynb
├── 2_data_preprocessing.ipynb
├── 3_aggregation.ipynb
├── 4_analysis.ipynb
├── 5_pca.ipynb
├── 6_modeling.ipynb
│
├── analysis_execution.py
│
├── utils/
│   ├── __init__.py
│   ├── anomaly_score.py
│   ├── event_analysis.py
│   ├── event_sampling.py
│   └── event_visualization.py
│
├── deprecated/
│   ├── d1_CMIP6_data_access.ipynb
│   └── d2_monsoon_data_decomposition.ipynb
│
├── .gitignore
└── README.md
```

The `deprecated/` directory contains earlier analysis notebooks that are retained for reference but are not part of the current workflow.

---

# Requirements

The analysis was developed in Python using scientific computing, geospatial and machine-learning libraries.

The main packages used throughout the repository include:

* Python
* Jupyter Notebook
* pandas
* NumPy
* xarray
* matplotlib
* Cartopy
* SciPy
* statsmodels
* scikit-learn
* XGBoost
* SHAP
* Hyperopt
* tqdm
* xcube

Additional packages are used for accessing external datasets and APIs.

A formal `requirements.txt` or `environment.yml` is currently not included in the repository.

---

# Data availability

The repository intentionally does **not** contain the large input datasets.

The `data/` directory is excluded through `.gitignore`.

The analysis requires, among other things:

* the Earth System Data Cube subset;
* the compound heat wave and drought event label cube;
* the compound-event summary dataset;
* the GLDAS land mask;
* the GLDAS dominant vegetation dataset;
* the vegetation-class translation table.

Some datasets are accessed remotely, while others are expected to be available locally.

Consequently, cloning this repository alone is **not sufficient to reproduce the complete analysis** without obtaining and preparing the required datasets.

---

# Running the analysis

The notebooks are numbered according to the conceptual development of the analysis:

```text
1. Data exploration
        ↓
2. Preprocessing / STL evaluation
        ↓
3. Single-event aggregation
        ↓
4. Complete event analysis
        ↓
5. Dimensionality reduction
        ↓
6. Machine learning
```

For the actual large-scale event processing, `analysis_execution.py` is the central script.

Before running it, the required datasets must be available at the paths expected by the script, including:

```text
data/esdc_subset.zarr
data/temp_zarr_extracted/mergedlabels_ranked_pot0.01_ne0.1_cmp_S1_T3_1950_2023.zarr
data/GLDASp5_landmask_025d.nc4
data/GLDASp5_domveg_NOAH3.6_025d.nc4
data/GLDA_veg_legend.csv
```

The output path is configured directly in `analysis_execution.py`.

---

# Important notes on reproducibility

This repository documents the computational workflow used for the Master's thesis. It is primarily intended as a **research code archive** rather than a ready-to-run software package.

In particular:

* large input datasets are not included;
* data paths are currently configured directly in the notebooks and scripts;
* no fixed Python environment is currently provided;
* some notebooks contain exploratory/development code;
* some intermediate output files referenced by the modelling notebooks are generated outside the repository.

For this reason, reproducing the exact thesis results requires access to the corresponding input datasets and the analysis environment used during the thesis.

---

# Author

**Georg Brandt**

M.Sc. Earth System Data Science and Remote Sensing
Leipzig University

Master's thesis, 2026

---

## Citation

If you use this repository or parts of the analysis workflow, please refer to the corresponding Master's thesis.

---

*This repository contains the computational workflow developed during my Master's thesis on vegetation responses to compound heat wave and drought events.*
