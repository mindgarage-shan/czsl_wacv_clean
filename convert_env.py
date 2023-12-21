import yaml

# Load environment.yml file
with open('environment.yml', 'r') as f:
    environment = yaml.load(f, Loader=yaml.SafeLoader)

# Extract packages and dependencies from environment.yml
dependencies = environment.get('dependencies', [])
requirements = []
for pkg in dependencies:
    if isinstance(pkg, str):
        # Handle package listed as a string with version and build
        pkg_name, pkg_version, pkg_build = pkg.split('=')[:3]
        pkg_name = pkg_name.strip()
        pkg_version = pkg_version.strip()
        pkg_build = pkg_build.strip()
        # if pkg_build:
        #     requirements.append('{}=={}+{}'.format(pkg_name, pkg_version, pkg_build))
        # else:
        requirements.append('{}=={}'.format(pkg_name, pkg_version))
    elif isinstance(pkg, dict):
        # Handle package listed as a dictionary
        if 'pip' in pkg:
            # Handle pip package with specified version
            for pip_pkg in pkg['pip']:
                requirements.append(pip_pkg)
        else:
            # Handle package with specified name, version, and build
            name = pkg.get('name', '')
            version = pkg.get('version', '')
            build = pkg.get('build', '')
            if name:
                if version and build:
                    # Split "package_name=version=build" into separate components
                    pkg_name = name.split('=')[0]
                    pkg_version = version.split('=')[0]
                    # Concatenate name, version, and build with "=="
                    requirements.append('{}=={}+{}'.format(pkg_name, pkg_version, build))
                elif version:
                    requirements.append('{}=={}'.format(name, version))
                else:
                    requirements.append(name)

# Write requirements.txt file
with open('requirements.txt', 'w') as f:
    for req in requirements:
        f.write(req + '\n')
