# Ingest Process
```{eval-rst}
.. py:currentmodule:: jeta.archive.ingest
```
## Public Ingest API 
```{eval-rst}
.. autofunction:: calculate_delta_times
```
```{eval-rst}
.. autofunction:: sort_msid_data_by_time
```
```{eval-rst}
.. autofunction:: move_archive_files
```
```{eval-rst}
.. autofunction:: execute
```

## Private Ingest Methods
```{eval-rst}
.. autofunction:: _process_csv
```

```{eval-rst}
.. autofunction:: _process_hdf
```
## Supported Ingest Modes

* Single MSID FOF (CSV) - Comma-delimited tabular data for a single msid in FOF format.
* Multiple MSIDs FOF (CSV) - Comma-delimited tabular data for a multiple msids in FOF format.
* HDF5 Multiple MSIDs - HDF5 files with sample data partitioned into datasets.

## Manually Starting an Ingest from the CLI
- TBD
## Ingest Scheduling

- TBD

## Telemetry Archive Structure

- See the LITA Systems Technical Documentation for acrhive implementation details.