#!/usr/bin/env bash

######## INIT PARAMS HERE #######
test_size=100
val_size=100
train_manifest_path=
########### END OF INIT #########

work_path=$(dirname ${train_manifest_path})
temp_train_manifest=${work_path}/temp_train_manifest.json

echo "Making validation split"
cat ${train_manifest_path} | head -n ${val_size} > ${work_path}/validation_manifest.json  || exit 1;
tail --lines=+$((val_size+1)) ${train_manifest_path} > ${temp_train_manifest}  || exit 1;

echo "Making test split"
cat ${temp_train_manifest} | head -n ${test_size} > ${work_path}/test_manifest.json  || exit 1;
tail --lines=+$((test_size+1)) ${temp_train_manifest} > ${train_manifest_path}  || exit 1;

rm -rf ${temp_train_manifest}  || exit 1;
echo "Done"
exit 0;
