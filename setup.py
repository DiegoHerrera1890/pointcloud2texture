from setuptools import setup, find_packages

setup(
    name="pointcloud2texture",
    version="0.1",
    packages=find_packages(where="src") + ["scripts"],  # Include scripts explicitly
    package_dir={"": "src", "scripts": "scripts"},  # Map scripts correctly
    install_requires=[
        "numpy",
        "Pillow",
        "scipy",
        "glog",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "run_texturing=scripts.run_texturing:main",
        ],
    },
)


