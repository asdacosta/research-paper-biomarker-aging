# Nonlinear and Marginally Coordinated Age-Related Transitions in Physiological Biomarkers

Code and analysis pipeline for a survey-weighted, cross-sectional study of six physiological biomarkers using NHANES 2017–2018 data.

The study examines age-related nonlinearity, clustering of estimated turning ages, and whether a coupled dynamical system can meaningfully represent relationships between population-level biomarker trajectories.

**Key findings**
- Survey weighting changed the nonlinearity classification of albumin and white blood cell count.
- The turning-age clustering test yielded *p* = 0.054, or *p* = 0.048 for the subset of more precisely estimated biomarkers.
- A two-biomarker dynamical model achieved \(R^2 = 1.00\), illustrating why a perfect fit to cross-sectional population trajectories does not establish an underlying dynamic relationship.

The manuscript is available at `paper/paper_v2.docx`.

## Reproducing the analysis

### 1. Download the NHANES data

Place these five files in `data/raw/`:

- [DEMO_J.XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT)
- [BIOPRO_J.XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.XPT)
- [CBC_J.XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.XPT)
- [HSCRP_J.XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HSCRP_J.XPT)
- [GLU_J.XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.XPT)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

Run the scripts in order from the `src/` directory:

```bash
python3 load_data.py
python3 curve_fit.py
python3 topic1_weighted.py
python3 topic2_weighted.py
python3 dynamics.py
```

The scripts merge and process the data, fit age trajectories, test nonlinearity and turning-age clustering, and fit the coupled dynamical model. The unweighted comparison scripts are retained for comparison with the weighted results.

## Repository structure

```text
data/
  raw/          # Downloaded NHANES files (not tracked)
  processed/    # Processed datasets
src/            # Analysis scripts and tests
figures/        # Generated figures
paper/          # Manuscript
```

## Validation and limitations

Code tests cover the data merge logic, breakpoint search on synthetic data, and weighted cluster-bootstrap procedures.

The study uses one NHANES cycle and estimates variance by resampling primary sampling units within strata rather than using Taylor series linearization. Turning ages are estimated using a single-hinge model over a bounded age range. These limitations are discussed in the manuscript.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the accompanying manuscript in `paper/`.
