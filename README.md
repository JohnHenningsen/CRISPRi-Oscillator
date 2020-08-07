Code used in "Single cell characterization of a synthetic bacterial clock with a hybrid feedback loop containing dCas9-sgRNA" <https://www.biorxiv.org/content/10.1101/2020.07.16.206722v1>

# Single Cell Analysis

- moma_analysis.py
    - function definitions

- extract_data.ipynb
    - input: folder with .tif for each selected gchannel; brightfield (BF), GFP (FI) and segmentation (from brightfield, ilastik) (SEG) in seperate folder
    - extract data from tifs (mother cell FI, area, background, ...), save into 1 .csv per gchannel
    - extra input: time vector extracted from metadata in seperate script
    - cleanup and correct fluorescence and time data and collect into single .csv
    
- figures.ipynb and SI_figures.ipynb
    - input: .csv output from extract_data.ipynb
    - compute data
    - create figures (cosmetic editing done in Illustrator)


# imageJ Scripts

- Use these scripts to process .nd2 microscopy stacks into tifs cropped to single growth channels, then segment with ilastik. 
- for normal nd2 files, preprocess_new works
- in case of a failed exposure during the timelapse, use the following scripts
    - for skipped timepoint nd2 files series don't seem to work
    - preprocess_specify and preprocess_split did not help. solution: open nd2 manually (virtual stack), use bio-formats exporter. use crop script afterwards
    - use fix_shift with respective time stamps to fix videos by splicing between positions
- gchannel_crop for semi automatic cropping of gchannels (select for regular growth in BF only)
- use channel splitting script to only use BF for ilastik

Note: Weird Windows bug: If Windows Explorer trys to create a thumbnail of a file that is currently being written by imageJ, the script crashes. Use list view (no thumbnails) or close explorer. 


# Simulation

Stochastic simulation. Reactions and rate constants described in the SI. 3 variants using different models to describe CRISPRi (hill function, irreversible binding, displacement by replication). 