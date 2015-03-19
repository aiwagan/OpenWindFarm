from distutils.core import setup
setup(name='OpenWindFarm',
      version='0.1',
      author='Ryan King',
      author_email='ryan.king@nrel.gov',
      packages = ['openwindfarm',
                  'openwindfarm.problems',
                  'openwindfarm.solvers',
                  'openwindfarm.domains',
                  'openwindfarm.functionals',
                  'openwindfarm.farm',
                  'openwindfarm.options',
                  'openwindfarm.turbines']
      )
