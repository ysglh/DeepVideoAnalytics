cython bbox.pyx
cython cython_nms.pyx
cython gpu_nms.pyx
python setup.py build_ext --inplace
mv lib/utils/* ./
rm -rf build
rm -rf lib

