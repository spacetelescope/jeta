CREATE TABLE archfiles (
  filename        text not null,
  filetime        int,
  year            int,
  doy             int,
  tstart          float not null,
  tstop           float not null,
  rowstart        int not null,
  rowstop         int not null,
  startmjf        int ,
  stopmjf         int ,
  date            text not null,

  CONSTRAINT pk_archfiles PRIMARY KEY (filename)
);

CREATE INDEX idx_archfiles_filetime ON archfiles (filetime);
