from setuptools import setup


APP = ['src/main.py']
DATA_FILES = [
    ("data/markers", ["data/markers/marker_1.png"]),
    ("data/marker_keypoints", ["data/marker_keypoints/marker_1_keypoints.npz"]),
    ("data/models", ["data/models/Cow.obj"]),
]
OPTIONS = {
    "plist": {
        "CFBundelName": "MyARProject",
        "CFBundelVersion": "1.0",
        "CFBundleIdentifier": "com.example.myarproject",
        "NSCameraUseContinuityCameraDeviceType": True,
        'NSCameraUsageDescription': 'This app requires camera access to detect markers and overlay AR elements.',
    },
    "packages": ["src/utils"],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app", "opencv-python", "matplotlib", "numpy"],
)
