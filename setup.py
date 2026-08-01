import os
import sys
from setuptools import setup, Extension, find_namespace_packages
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
            f'-DPROJECT_ROOT_DIR={project_root}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]
        
        # --- NEW: Check environment variable for Vulkan build ---
        # This is set by setup.bat or setup.sh
        build_vulkan = os.environ.get('HIERARCHOS_BUILD_VULKAN', 'OFF').upper()
        if build_vulkan in ('ON', 'TRUE', '1'):
            print("--- Enabling Vulkan build ---")
            cmake_args.append('-DHIERARCHOS_BUILD_VULKAN=ON')
        else:
            print("--- Disabling Vulkan build (CPU only) ---")
            cmake_args.append('-DHIERARCHOS_BUILD_VULKAN=OFF')
        # --- END NEW ---
        
        cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")

        build_args = []
        
        if self.compiler.compiler_type == "msvc":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j' + str(os.cpu_count() or 1)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# --- Main Setup ---
setup(
    name='hierarchos_matmul',
    version='0.21.0',
    author='Makhi Burroughs', # Filled in
    author_email='saltpepper312@gmail.com', # Placeholder
    description='Custom C++ kernel with optional Vulkan support for the Hierarchos project',
    long_description='',
    # Include the modular Python runtime as well as the optional native
    # accelerator. Most subdirectories intentionally use namespace-package
    # layout (no __init__.py), so find_packages() would still omit them.
    packages=find_namespace_packages(include=["hierarchos", "hierarchos.*"]),
    ext_modules=[CMakeExtension('hierarchos_matmul', sourcedir='cpp')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    # The active modular runtime uses PEP 604 union annotations (``T | None``),
    # which are syntax only on Python 3.10+.  Advertising 3.8/3.9 lets an
    # installation succeed and then fail while importing the model.
    python_requires=">=3.10",
)
