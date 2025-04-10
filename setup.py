from setuptools import setup, find_packages

setup(
    name='wobert_phy',
    package_dir={"": "src"},
    packages=find_packages("src"),
    version='0.0.1',
    license='MIT',
    description='wobert_phy_pytorch',
    author='Tian Zhengshuyuan',
    author_email='298208490@qq.com',
    url='https://github.com/Tianzhengshuyuan/WoBERT_PHY_pytorch',
    keywords=['wobert', 'pytorch', 'phy'],
    install_requires=['transformers'],
)