from setuptools import find_packages, setup


package_name = "cuda_nav_bringup"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/cudanav_closed_loop.launch.py",
                "launch/cudanav_recorded_shadow.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/controller.yaml",
                "config/controller_recorded_shadow.yaml",
                "config/rosbag_qos_overrides.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CudaRobotics maintainers",
    maintainer_email="rsasaki0109@users.noreply.github.com",
    description="Deterministic closed-loop bringup and simulation for CudaNav",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cudanav_loopback_simulator = "
            "cuda_nav_bringup.loopback_simulator:main",
            "lifecycle_orchestrator = "
            "cuda_nav_bringup.lifecycle_orchestrator:main",
            "follow_path_mission = "
            "cuda_nav_bringup.follow_path_mission:main",
            "recorded_path_shadow = "
            "cuda_nav_bringup.recorded_path_shadow:main",
        ],
    },
)
