echo "1socket"
python compare_result.py -f 1socket_with_fusion.log -f 1socket_no_fusion.log
echo "1thread"
python compare_result.py -f 1thread_with_fusion.log -f 1thread_no_fusion.log
echo "4thread"
python compare_result.py -f 4thread_with_fusion.log -f 4thread_no_fusion.log
