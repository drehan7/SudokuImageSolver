#!/bin/bash

run=false
force=false
test=false

log=build.log

if [ -z $log ]; then
    touch $log
fi

buildtest() {
    make test &> $log
    echo $?
}

while getopts rft flag
do
    case "$flag" in
        f) force=true;;
        r) run=true;;
        t) test=true;;
        *) echo "invalid arg";;
    esac
done

success=false
ret=`buildtest`
if [ $ret -eq 0 ]; then
    echo "Success"
    success=true
else
    echo "Failed with $ret"
    cat $log
    exit 1
fi

if [ $test = true ]; then
    echo "Running tests..."
    buildtest
    testret=$?
    if [ $testret -eq 0 ]; then
        ./test
    fi
fi

