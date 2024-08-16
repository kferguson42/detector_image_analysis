# detector_image_analysis
This repository contains code designed to analyze optical images of cryogenic SPT detectors. It can handle both SPT-3G TES bolometers as well as SPT_3G+ MKIDs. The TES pipeline uses visual features determined from the images as inputs to random forest algorithms designed to predict the operatability and performance characteristics of the bolometers. The MKID pipeline eschews the machine learning in favor of a more directed search for defects in the detectors.

NOTE: This is the public version of this repository. Because the code deals with proprietary hardware designs, certain auxillary files containing this design info have been omitted, as well as certain auxillary scripts that were written to convert the data into the format expected by the main scripts described below. Without these (as well as the actual detector images), the main scripts are not exactly runnable; this public-facing version is meant simply as a demonstration of some of the methods I've implemented.

## SPT-3G TES bolometers
The code for this detector geometry lives in the `tes_analysis` sub-directory. The main scripts to be aware of are:
* `image_features.py`, which takes detector images as an input and returns as an output information on the detectors such as their area and perimeter, surface roughness, etc.
* `do_machine_learning_<classification/regression>.py`, which takes the visual and cryogenic features as inputs and saves out a pickle file of the relevant ML outputs.
Beyond this there are a number of auxillary scripts to reformat data in the way the ML scripts expect it, to turn detector design GDS files into usable pickle data products, to plot figures, etc.

## SPT-3G+ MKIDs
This code all lives in the `mkid_analysis` sub-directory (though it occasionally imports modules that live in the top-level directory). This iteration of the project dropped the ML in favor of predicting cryogenic performance straight from the images themselves. Two main scripts to be aware of:
* `stitch_images.py`, which takes the individual image tiles direct from the microscope and stitches them together into one single image. Stitching is necessary to capture the entire area of the detector while keeping a high enough resolution to actually perform our analysis.
  * NOTE: This nominally makes use of the external package `m2stitch`, a Python implementation of the MIST stitching algorithm. For our particular detector geometry, the default MIST algorithm led to occasional stitching errors, so I made some modifications. You can find my public fork of `m2stitch` [here](https://github.com/kferguson42/m2stitch_fork).
* `analyze_mkid_pixel.py`, which does the main image analysis. It determines the location and orientation of the detectors in the image, searches for defects using a flood fill algorithm, and measures the width of the detector conducting lines, all while generating plots about what it finds.
There are also some other scripts providing auxillary functionality such as converting GDS design information into a script-usable format or creating images with simulated defects.

# Installation
Nothing needs to be installed _per se_ to run this code. However, this repository makes use of a number of custom modules. In order for these imports to succeed, you must add the location of this repository in your directory structure to your personal `PYTHONPATH` environment variable. This can be done, for example, by adding the line `export PYTHONPATH="${PYTHONPATH}:/path/to/repository"` to your `.bashrc` or `.bash_profile` (whichever one you set your environment variables in).

Note also that some functions in the `geometry_utils.py` module make use of the [shapely](https://pypi.org/project/shapely/) package, which you'll need to install to use those functions.
