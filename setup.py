from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['matplotlib','tensorflow','Pillow']
setup(name='trainer',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      description='Cycle GAN',
      author='Mridul Gupta',
      author_email='mridulgupta9@gmail.com',
      license='free',
      zip_safe=False)