from setuptools import setup
from torch.utils import cpp_extension

setup(name='bev_pool_forward',
      ext_modules=[cpp_extension.CUDAExtension(name='bev_pool_forward', 
                                               sources=['src/bev_pool.cpp', 'src/bev_pool_cuda.cu'],
                                               extra_compile_args={'cxx':['-g']
                                                                   })],
      license='Apache License v2.0',
      cmdclass={'build_ext': cpp_extension.BuildExtension})