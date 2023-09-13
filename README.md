# xnat-report-fmriprep-group

## Description
This project generates quality reports for pre-processed fMRI datasets following the BIDS standard. It uses metrics such as Framewise Displacement (FD) and sudden signal variations (DVARS) to assess data quality.

## Project structure
* quality.py: Contains functions for extracting metrics and generating figures.
* test.py: An example script showing how to use the functions in quality.py.
* bootstrap-fmriprep.yml: A YAML configuration file for report layout and style. It respects the template proposed by nireport, which you can find [here.](https://github.com/nipreps/nireports/blob/main/nireports/assembler/report.py)

## Dependencies
* pandas
* matplotlib
* plotly
* uuid
* os
* pathlib
import argparse

## File quality.py
The quality.py file is the heart of this project. It contains several functions that facilitate the processing and analysis of fMRI data in compliance with BIDS standards. Here's an overview of the main functions and their uses:

### Main functions
* extract_unique_tasks(all_tables)

Description: This function takes as input a list of tables containing fMRI metrics and returns a list of unique tasks present in these tables.

* generate_figure(all_tables, repetition_times, signal, output_dir)

Description: Generate Plotly figures to visualize a specified physiological signal across multiple tasks and subjects.
The function generates two types of figures: one for each unique task and one for all tasks.
Each figure has dropdown options to toggle visibility of data for each subject.
The color of the traces in the 'all tasks' figure depends on the task.
The function saves the figures in HTML format in the specified output directory.

* generate_figures_motion(all_tables, repetition_times, signals, output_dir)

Description: generate_figures_motion function in the code is designed to generate Plotly figures for motion parameters in neuroimaging data.

* display_outliers(all_tables, repetition_times, output_dir, fd_threshold=0.5, dvars_threshold=1.5)

Description: This function generates graphs to display outliers based on Framewise Displacement (FD) and DVARS thresholds.



### Other functions

* get_bids_data(data_path)
Description: This function extracts BIDS tables and entities from the data access path.

* generate_report_with_plots(...)
Description: This function generates a final HTML report compiling all figures and metrics.

## Usage

### Bash command line
Use the test.py script to generate reports:

```bash
python test.py /path/to/data --fd_threshold 0.5 --dvars_threshold 1.5
```
This will create a report in the /path/to/data/report folder.


### Docker

This project is designed to run in a Docker container, which ensures that all dependencies and the runtime environment are uniformly configured.

Build the container with for example 
```docker
docker build -t report .
```

Once you've built the Docker image, you can run the container using the following command:
```docker
docker run \
-v /path/to/data:/in/ \
-v /path/to/data:/out/ \
report \
python3 /app/test.py --fd_threshold 0.5 --dvars_threshold 1.5 /in/
```



## Parameters
* data_path: Path to BIDS data folder.
* --fd_threshold: FD threshold (default = 0.5).
* --dvars_threshold: DVARS threshold (default = 1.5).

Reports will be saved in report/reportlet/figures.