
CREATE TABLE IF NOT EXISTS copy_ingest_history (
  ingest_id         text not null, -- unique id for each ingest
  discovered_files  int, -- the number of files discovered for ingest
  processed_files  int, -- the number of files discovered for ingest
  coverage_start    float not null,
  coverage_end      float not null,
  tstart            int not null, -- start of ingest
  tstop             int not null, -- end of ingest
  rowstart          int,
  ingest_status     text, -- success, failure, null
  new_msids         int, -- number of new msids added during this ingest
  chunk_size        int, -- processing chunks used during this ingest

  CONSTRAINT pk_ingest_id PRIMARY KEY (ingest_id)
);

CREATE TABLE IF NOT EXISTS copy_archfiles (
  filename                 text not null, -- filename for individual the ingest file
  year                       int, -- year file was created
  doy                        int, -- day of year file was created
  tstart                     float not null, -- start of data coverage for the file, in unix time
  tstop                      float not null, -- end of data coverage for the file,unix time
  offset                     int not null, -- offset used
  chunk_group                int not null, -- the chunk group to which this file was included in for an ingest
  startmjf                   int,
  stopmjf                    int,
  processing_date            text not null,
  ingest_id                  text not null,

  FOREIGN KEY(ingest_id) REFERENCES ingest_history(ingest_id),
  CONSTRAINT pk_archfiles PRIMARY KEY (filename)
);

CREATE INDEX IF NOT EXISTS idx_archfiles_tstart ON archfiles (tstart);

INSERT INTO copy_ingest_history (ingest_id,  discovered_files,  processed_files,  coverage_start, coverage_end, tstart,  tstop,  rowstart, ingest_status,  new_msids,  chunk_size)
	SELECT  cast (ingest_id as text),  discovered_files,  processed_files,  coverage_start, coverage_end, tstart,  tstop,  rowstart, ingest_status,  new_msids,  chunk_size FROM ingest_history;

INSERT INTO copy_archfiles  (filename, year, doy,  tstart, tstop, "offset", chunk_group, startmjf, stopmjf, processing_date, ingest_id)
	SELECT  filename, year, doy,  tstart, tstop, "offset", chunk_group, startmjf, stopmjf, processing_date,  cast (ingest_id as text)  FROM archfiles ;