[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgithub.com%2FOsloMag%2FOmag/HEAD?urlpath=%2Fdoc%2Ftree%2Fmag_fits.ipynb)

This comprises a set of notebooks enabling the interactive analysis of paleomagnetic directional data collected during stepwise demagnetization.
mag_fits.ipynb provides tools to visualize sample demagnetization data and to fit lines and planes to these data.
mag_stats.ipynb provides tools to visualize and perform statistical characterizations of collections of line and plane fits on paleomagnetic data.
The remainder of the files hold the methods supporting these two notebooks:
-fitsUI.py and statsUI.py control the user interfaces of the respective notebooks
-processing.py contains the mathematical and statistical methods needed
-plotting.py contains the methods setuping up and displaying the plots
