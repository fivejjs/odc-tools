from setuptools import setup

setup(
    name="odc_geom",
    use_scm_version={"root": "../..", "relative_to": __file__},
    setup_requires=["setuptools_scm"],
    author="Open Data Cube",
    author_email="",
    maintainer="Open Data Cube",
    description="Miscellaneous Geometry helper methods",
    long_description="",
    license="Apache License 2.0",
    tests_require=["pytest"],
    install_requires=[
        "affine",
    ],
    packages=["odc.geom"],
    zip_safe=False,
)
