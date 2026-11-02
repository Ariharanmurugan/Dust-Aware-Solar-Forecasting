# Dataset

The dataset file is not kept in this repository because of its size. This page records what it
contains, how it was built and how to check a copy.

## The file

| | |
|---|---|
| Name | `uae_solar_dust_hourly_2015_20260731.csv` |
| Size | 304,560 rows x 26 columns, 44.8 MB |
| SHA-256 | `154dcf9d01384bc30bd7a21522288421c1e4ef64aa95f5285d5a6d949b94de5a` |
| Period | 1 January 2015 to 31 July 2026, hourly, UTC |
| Sites | Abu Dhabi (24.45 N, 54.37 E), Dubai (25.20 N, 55.27 E), Al Ain (24.21 N, 55.74 E) |
| Rows per site | 101,520 consecutive hours, no gap and no duplicate |

The notebook checks the first 16 characters of the fingerprint before it reads the file, so a
different or partial copy stops the run rather than producing quiet nonsense.

## Source

NASA POWER hourly API, community RE, downloaded on 18 October 2026. Seventeen parameters:

| Group | Columns | Product |
|---|---|---|
| Irradiance and sun | all-sky GHI, clear-sky GHI, DNI, DHI, clearness index, solar zenith angle, cloud amount | CERES SYN1deg |
| Dust | AOD at 550 nm, AOD at 840 nm | CERES SYN1deg |
| Weather | 2 m temperature, dew point, relative humidity, specific humidity, 10 m wind speed, 10 m wind direction, surface pressure, precipitation | MERRA-2 |

The remaining columns are the site name and the UTC timestamp.

Note on attribution: NASA POWER's own file headers attribute the irradiance, clearness, solar
zenith angle, cloud amount and both AOD columns to CERES SYN1deg. Only the eight weather
variables come from MERRA-2.

## Getting a copy

Request the three sites and the parameters above from the NASA POWER hourly API
(<https://power.larc.nasa.gov/>), replace the fill value -999 with an empty cell, and concatenate
the sites into one CSV sorted by site and then by hour.

A copy can be verified with:

```bash
sha256sum uae_solar_dust_hourly_2015_20260731.csv
```

A download made today runs to the latest available hour, so the row count and the fingerprint will
differ from the file above, and the notebook's checks would need updating. Use the original file to
reproduce the published results.

## Where to put it

The notebook looks for the dataset in this order:

1. the path in the `DATA_FILE` environment variable;
2. `data/uae_solar_dust_hourly_2015_20260731.csv` in this repository;
3. the same file inside `data/uae_solar_dust_hourly_2015_20260731.zip`, which is unzipped once;
4. Google Drive, when the notebook runs in Colab and the file is kept there.

## Acknowledgement

These data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through
the NASA Earth Science/Applied Science Program. The data carry NASA POWER's own terms, not the MIT
licence that covers the code in this repository.
