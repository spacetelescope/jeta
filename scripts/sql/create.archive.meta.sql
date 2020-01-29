CREATE TABLE IF NOT EXISTS initialized_mnemonics (

  mnemonic        text not null,
  initialized     int,

  CONSTRAINT pk_initialized_mnemonics PRIMARY KEY (mnemonic)
);

CREATE TABLE IF NOT EXISTS archfiles (
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

CREATE INDEX IF NOT EXISTS idx_archfiles_filetime ON archfiles (filetime);