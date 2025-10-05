import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

# --- Get Pybind11 and Python headers ---
try:
    import pybind11
except ImportError:
    print("Error: pybind11 is required to build the C++ kernel. Please install it with 'pip install pybind11'")
    sys.exit(1)

# --- CMake Build Extension Class ---
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extensions(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed")

        if self.compiler is None:
            super().run()
            self.compiler = None
        
        if self.compiler is None:
            from distutils.ccompiler import new_compiler
            self.compiler = new_compiler(compiler=self.compiler, dry_run=self.dry_run, force=self.force)
            self.compiler.initialize()

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Get the project root directory
        project_root = os.path.abspath(os.path.dirname(__file__))
        
        # This is the directory where the final library will be installed by pip
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            # --- KEY CHANGE: Pass the project root to CMake ---
            f'-DPROJECT_ROOT_DIR={project_root}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]
        
        cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")

        build_args = []
        
        if self.compiler.compiler_type == "msvc":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j' + str(os.cpu_count())]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# --- Main Setup ---
setup(
    name='chronos_matmul',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Custom C++ kernel for the Chronos project',
    long_description='',
    ext_modules=[CMakeExtension('chronos_matmul', sourcedir='cpp')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
