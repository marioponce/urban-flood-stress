# Urban Flood Stress

Modeling flood-related infrastructure stress and institutional response in New York City using geospatial data and machine learning.

---

## Overview

Urban flooding affects infrastructure systems, mobility, and public services in complex and spatially heterogeneous ways. This project investigates how flood-related stress propagates across urban infrastructure and how institutional response patterns vary across neighborhoods and socioeconomic conditions.

The project integrates geospatial datasets, urban infrastructure information, flood-related service requests, and machine learning methods to identify patterns of vulnerability, exposure, and response.

---

## Research Objectives

- Quantify flood-related infrastructure stress in NYC
- Analyze institutional response dynamics using 311 service requests
- Identify spatial inequalities in exposure and response
- Evaluate the role of road-network structure and infrastructure criticality
- Develop interpretable machine learning models for flood-related urban analysis

---

## Data Sources

### Public datasets

- NYC 311 Service Requests
- USGS 3DEP Elevation Data
- NYC Digital City Map (DCM)
- U.S. Census ACS
- NOAA precipitation and environmental datasets (optional)

---

## Methodology

The workflow combines:

- Geospatial data processing
- Spatial joins and feature engineering
- Network analysis
- Machine learning models
- Temporal validation strategies
- Infrastructure criticality metrics
- Spatial equity analysis

Potential models include:

- Random Forest
- Logistic Regression
- Gradient Boosting
- Spatial and temporal clustering approaches

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
├── figures/
├── results/
├── docs/
└── README.md