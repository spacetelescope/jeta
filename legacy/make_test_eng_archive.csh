# From within the hg eng_archive repo
# source make_test_eng_archive.csh

setenv ENG_ARCHIVE $PWD/test/eng_archive
mkdir -p test/eng_archive
rm -rf test/eng_archive/*

rm -f test/make_eng_archive.log
touch test/make_eng_archive.log

echo "Making regr data..."
./make_regr_data.py --start 2010:260 --stop 2010:270 --data-root test/eng_archive >>& test/make_eng_archive.log

echo "Tarring..."
cd test
tar zcf eng_archive.tar.gz eng_archive
cd ..

echo "Updating archive..."
./update_archive.py --date-now 2010:271 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:272 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:273 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:274 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:275 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:276 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:277 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:278 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:279 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:280 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:281 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:282 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:283 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:284 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:285 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:286 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:287 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:288 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:289 --data-root test/eng_archive >>& test/make_eng_archive.log
./update_archive.py --date-now 2010:290 --data-root test/eng_archive >>& test/make_eng_archive.log



