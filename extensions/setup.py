"""
这个是用来设置cuda环境的
Python包管理工具setuptools,是Python distutils增强版的集合,可以帮助我们更简单的创建和分发Python包，尤其是拥有依赖关系的
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules=[
    CUDAExtension('dapalib_light', [
        'association.cpp',
        'gpu/nmsBase.cu',
        'gpu/bodyPartConnectorBase.cu',
        'gpu/cuda_cal.cu',
        ], 
        include_dirs=['/usr/local/cuda/include', '/usr/local/lib'] ,   # '/usr/include/eigen3'
    ),         
]

setup(
    name='dapalib',
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
