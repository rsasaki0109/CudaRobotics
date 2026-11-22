from setuptools import find_packages, setup


package_name = "cuda_nav_tracking"

setup(
    name=package_name,
    version="0.4.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="CudaRobotics maintainers",
    maintainer_email="rsasaki0109@users.noreply.github.com",
    description="Scan-based moving-object tracker publishing TrackedObstacleArray",
    license="MIT",
    entry_points={
        "console_scripts": [
            "scan_tracker_node = cuda_nav_tracking.scan_tracker_node:main",
        ],
    },
)
